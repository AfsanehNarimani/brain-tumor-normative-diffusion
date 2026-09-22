"""Exact discrete receptive-field interval propagation for the E1-C encoder."""

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class RFGeometry:
    input_size: int
    output_size: int
    effective_stride: int
    theoretical_rf_size: int
    theoretical_intervals: tuple[tuple[int, int], ...]
    clipped_intervals: tuple[tuple[int, int], ...]


def _scalar(value, name):
    if isinstance(value, tuple):
        if len(set(value)) != 1:
            raise ValueError(f'Non-isotropic {name}: {value}')
        value = value[0]
    return int(value)


def encoder_1d_layers(encoder):
    """Read Conv3d/MaxPool3d geometry from the actual Encoder modules."""
    layers = []
    for index, block in enumerate(encoder.blocks):
        if index:
            pool = encoder.pool
            layers.append(('pool', _scalar(pool.kernel_size, 'pool kernel'),
                           _scalar(pool.stride, 'pool stride'),
                           _scalar(pool.padding, 'pool padding')))
        for module in block:
            if isinstance(module, nn.Conv3d):
                layers.append(('conv', _scalar(module.kernel_size, 'conv kernel'),
                               _scalar(module.stride, 'conv stride'),
                               _scalar(module.padding, 'conv padding')))
    return tuple(layers)


def derive_geometry(input_size, output_size, layers):
    if input_size < 1 or output_size < 1:
        raise ValueError('Input and output sizes must be positive')
    jump, rf, start = 1, 1, 0
    for kind, kernel, stride, padding in layers:
        if kind not in ('conv', 'pool') or kernel < 1 or stride < 1 or padding < 0:
            raise ValueError(f'Invalid layer geometry: {(kind, kernel, stride, padding)}')
        start -= padding * jump
        rf += (kernel - 1) * jump
        jump *= stride
    if jump != (input_size // output_size):
        raise ValueError(f'Geometry stride {jump} does not match sizes {input_size}->{output_size}')
    if start + (output_size - 1) * jump + rf <= 0:
        raise ValueError('Derived intervals do not overlap the input')
    theoretical = tuple((start + index * jump, start + index * jump + rf)
                        for index in range(output_size))
    clipped = tuple((max(0, lower), min(input_size, upper)) for lower, upper in theoretical)
    return RFGeometry(input_size, output_size, jump, rf, theoretical, clipped)


def derive_encoder_geometry(encoder, input_size=96, output_size=12):
    return derive_geometry(input_size, output_size, encoder_1d_layers(encoder))
