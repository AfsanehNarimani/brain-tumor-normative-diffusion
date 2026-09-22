"""Streaming population moments and exact disk-backed float32 percentiles for D0-A."""
import csv
import tempfile
from pathlib import Path

import numpy as np

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
PERCENTILE_METHOD = (
    'Exact linear percentiles (NumPy method=linear convention). Finite float32 values '
    'are stored as order-preserving uint32 keys in a temporary channel-major disk file; '
    'each channel is sorted in place. Global order statistics use 32-step integer '
    'rank searches across sorted channels, without concatenating latent arrays. '
    'Interpolation and moments use float64. No sampling or approximation. '
    'Scratch disk: case_count * 10 * product(latent_shape) * 4 bytes; '
    'No full-file mapping: sort/read maps at most one channel at a time. RAM working arrays are bounded by ten patches of one case plus float64 temporaries; '
    'mapped pages are managed by the OS. Scratch storage is deleted on exit.'
)


class Moments:
    """Chan's parallel variance update with float64 batch-centered sums of squares."""
    def __init__(self, channels=1):
        self.count = 0
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)
        self.minimum = np.full(channels, np.inf)
        self.maximum = np.full(channels, -np.inf)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError('Expected finite nonempty [channels,elements] values')
        n = values.shape[1]
        mean = values.mean(axis=1)
        delta = mean - self.mean
        total = self.count + n
        self.m2 += ((values - mean[:, None]) ** 2).sum(axis=1) + delta ** 2 * (self.count * n / total)
        self.mean += delta * (n / total)
        self.minimum = np.minimum(self.minimum, values.min(axis=1))
        self.maximum = np.maximum(self.maximum, values.max(axis=1))
        self.count = total

    def row(self, channel=0):
        return dict(count=self.count, mean=float(self.mean[channel]),
                    std=float(np.sqrt(self.m2[channel] / self.count)),
                    minimum=float(self.minimum[channel]), maximum=float(self.maximum[channel]))


def summarize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    moments = Moments()
    moments.update(values[None, :])
    return {**moments.row(), **percentile_fields(np.percentile(values, PERCENTILES, method='linear'))}


def percentile_fields(values):
    return {f'p{p:02d}': float(value) for p, value in zip(PERCENTILES, values)}


def ordered_keys(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return np.where(bits & np.uint32(0x80000000), ~bits, bits ^ np.uint32(0x80000000))


def key_values(keys):
    keys = np.asarray(keys, dtype=np.uint32)
    bits = np.where(keys & np.uint32(0x80000000), keys ^ np.uint32(0x80000000), ~keys)
    return bits.view(np.float32).astype(np.float64)


def exact_percentiles(sorted_channels, channel=None):
    """Select exact float32 order statistics with linear interpolation in float64."""
    size = sorted_channels.shape[1] if channel is not None else sorted_channels.size
    positions = np.asarray(PERCENTILES, dtype=np.float64) / 100 * (size - 1)
    ranks = np.concatenate((np.floor(positions), np.ceil(positions))).astype(np.int64)
    if channel is not None:
        keys = sorted_channels[channel, ranks]
    else:
        low = np.zeros(len(ranks), dtype=np.int64)
        high = np.full(len(ranks), 2**32 - 1, dtype=np.int64)
        # Search for the smallest ordered key with cumulative count > zero-based rank.
        for _ in range(32):
            mid = (low + high) // 2
            counts = np.zeros(len(ranks), dtype=np.int64)
            for row in sorted_channels:
                counts += np.searchsorted(row, mid.astype(np.uint32), side='right')
            below = counts <= ranks
            low = np.where(below, mid + 1, low)
            high = np.where(below, high, mid)
        keys = low.astype(np.uint32)
    lower, upper = np.split(key_values(keys), 2)
    weight = positions - np.floor(positions)
    delta = upper - lower
    return percentile_fields(np.where(weight >= 0.5, upper - delta * (1 - weight), lower + delta * weight))


class DiskKeys:
    """One raw file, explicit case-block writes, at most one channel mapped at a time."""
    def __init__(self, path, channels, elements_per_channel):
        self.path = Path(path)
        self.shape = (channels, elements_per_channel)
        self.size = channels * elements_per_channel
        with self.path.open('wb') as stream:
            stream.truncate(self.size * 4)

    def write_case(self, case_index, values):
        # values: [patch, channel, spatial...]; one bounded ten-patch case only.
        flat = np.moveaxis(values, 1, 0).reshape(self.shape[0], -1)
        with self.path.open('r+b') as stream:
            for channel, row in enumerate(flat):
                stream.seek((channel * self.shape[1] + case_index * row.size) * 4)
                stream.write(ordered_keys(row).tobytes())

    def mapped_channel(self, channel, mode='r'):
        return np.memmap(self.path, mode=mode, dtype=np.uint32,
                         offset=channel * self.shape[1] * 4, shape=(self.shape[1],))

    def sort(self):
        for channel in range(self.shape[0]):
            row = self.mapped_channel(channel, 'r+')
            try:
                row.sort(kind='quicksort')
                row.flush()
            finally:
                row._mmap.close()

    def __iter__(self):
        for channel in range(self.shape[0]):
            row = self.mapped_channel(channel)
            try:
                yield row
            finally:
                row._mmap.close()

    def __getitem__(self, item):
        channel, ranks = item
        row = self.mapped_channel(channel)
        try:
            return np.array(row[ranks], copy=True)
        finally:
            row._mmap.close()


PATCHES_PER_CASE = 10
FOREGROUND_PATCHES = 7


def patch_type(index):
    if not 0 <= index < PATCHES_PER_CASE:
        raise ValueError('Patch index must be in [0,9]')
    return 'foreground' if index < FOREGROUND_PATCHES else 'random'


def distribution_row(values):
    values = np.asarray(values, dtype=np.float64)
    result = summarize(values)
    result['latent_element_count'] = result.pop('count')
    result.update(RMS=float(np.sqrt(np.mean(values ** 2))),
                  mean_absolute_value=float(np.mean(np.abs(values))))
    return result


def audit_latents(samples, case_ids, output, latent_shape=(256, 12, 12, 12)):
    """Consume (patch metadata, latent); aggregate ten patches per biological case."""
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError('Expected nonempty unique biological case IDs')
    output = Path(output)
    channels, voxels = latent_shape[0], int(np.prod(latent_shape[1:]))
    channel_moments, global_moments = Moments(channels), Moments()
    rms_values, std_values = [], []
    iterator = iter(samples)
    with tempfile.TemporaryDirectory(prefix='.d0-scratch-', dir=output) as scratch, \
            (output / 'latent_case_statistics.csv').open('w', newline='') as case_stream, \
            (output / 'latent_patch_statistics.csv').open('w', newline='') as patch_stream:
        keys = DiskKeys(Path(scratch) / 'keys.bin', channels, len(case_ids) * PATCHES_PER_CASE * voxels)
        case_writer = patch_writer = None
        for case_index, case_id in enumerate(case_ids):
            case_values = np.empty((PATCHES_PER_CASE, *latent_shape), dtype=np.float32)
            for index in range(PATCHES_PER_CASE):
                try:
                    metadata, latent = next(iterator)
                except StopIteration as error:
                    raise ValueError('Missing latent patches') from error
                if (metadata['case_id'] != case_id or metadata['patch_index_within_case'] != index
                        or metadata['patch_type'] != patch_type(index)):
                    raise ValueError('Latent patch identity/order/type mismatch')
                latent = np.asarray(latent)
                if latent.shape != tuple(latent_shape):
                    raise ValueError(f'Unexpected latent shape: {latent.shape}; expected {latent_shape}')
                if latent.dtype != np.float32:
                    raise ValueError('Audit requires float32 latents; no silent precision conversion')
                if not np.isfinite(latent).all():
                    raise ValueError(f'Non-finite latent: {case_id} patch {index}')
                case_values[index] = latent
                channel_moments.update(latent.reshape(channels, voxels))
                global_moments.update(latent.reshape(1, -1))
                row = dict(case_id=case_id, patch_index_within_case=index,
                           patch_type=patch_type(index), **distribution_row(latent))
                if patch_writer is None:
                    patch_writer = csv.DictWriter(patch_stream, fieldnames=list(row))
                    patch_writer.writeheader()
                patch_writer.writerow(row)
            keys.write_case(case_index, case_values)
            row = dict(case_id=case_id, patch_count=PATCHES_PER_CASE, **distribution_row(case_values))
            if case_writer is None:
                case_writer = csv.DictWriter(case_stream, fieldnames=list(row))
                case_writer.writeheader()
            case_writer.writerow(row)
            rms_values.append(row['RMS'])
            std_values.append(row['std'])
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError('Unexpected extra latent patches')
        del case_values
        keys.sort()
        global_stats = {**global_moments.row(), **exact_percentiles(keys)}
        with (output / 'latent_channel_statistics.csv').open('w', newline='') as stream:
            writer = None
            for channel in range(channels):
                row = dict(channel_index=channel, **channel_moments.row(channel),
                           **exact_percentiles(keys, channel))
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
        return global_stats, dict(case_count=len(case_ids), total_patch_count=len(case_ids) * PATCHES_PER_CASE,
                                  per_case_RMS=summarize(rms_values),
                                  per_case_standard_deviation=summarize(std_values))
