"""Machine-readable research artifacts without changing training state."""
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch
import yaml


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def environment_metadata(parameter_count):
    def optional(function):
        try:
            return function()
        except Exception:
            return None
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1],
                                       stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    return dict(python_version=platform.python_version(), pytorch_version=str(torch.__version__),
                cuda_version=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                gpu_name=optional(lambda: torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
                mps_available=optional(lambda: torch.backends.mps.is_available()),
                platform=platform.platform(), hostname=optional(socket.gethostname),
                trainable_parameter_count=parameter_count, timestamp=datetime.now(timezone.utc).isoformat(),
                git_commit=optional(lambda: git('rev-parse', 'HEAD')),
                git_dirty=optional(lambda: bool(git('status', '--porcelain'))))


def snapshot(output, config, parameter_count, resume_epoch=None):
    """Keep previous config/environment versions on compatible resume."""
    output = Path(output)
    path = output / 'config.yaml'
    if path.exists():
        if resume_epoch is None:
            raise ValueError('Experiment config already exists; use resume or a fresh output directory.')
        previous = yaml.safe_load(path.read_text())
        for key in ('data', 'model', 'debug', 'selection_metric'):
            if previous.get(key) != config.get(key):
                raise ValueError(f'Incompatible config snapshot: {key}')
        allowed = {'epochs', 'device', 'output_dir', 'resume', 'num_workers', 'validation_frequency'}
        for key in set(previous['train']) | set(config['train']):
            if key not in allowed and previous['train'].get(key) != config['train'].get(key):
                raise ValueError(f'Incompatible config snapshot: {key}')
        archive = output / f'config.before_epoch_{resume_epoch + 1}.yaml'
        if archive.exists():
            raise ValueError(f'Config archive already exists: {archive}')
        archive.write_text(path.read_text())
        env = output / 'environment.json'
        if env.exists():
            (output / f'environment.before_epoch_{resume_epoch + 1}.json').write_text(env.read_text())
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(output / 'environment.json', environment_metadata(parameter_count))
