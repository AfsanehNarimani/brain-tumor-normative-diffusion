"""Official BraTS-evaluation Python API with unchanged bundled gli config.

No local metric formulas, legacy evaluator, or custom-metric fallback.
"""
import hashlib
import importlib
from importlib.metadata import distribution, version, PackageNotFoundError
import json
import math
import platform
import yaml
from numbers import Real
from pathlib import Path
import subprocess

REGIONS = ('NETC', 'SNFH', 'ET', 'RC', 'WT', 'TC')
REPOSITORY = 'https://github.com/BraTS/BraTS_evaluation'
TERMINOLOGY = ('BraTS-compatible local evaluation using the official BraTS evaluation '
               'package with the bundled gli configuration')


def source_git_metadata(package):
    """Null metadata for wheel installs; inspect only an identifiable upstream checkout."""
    result = dict(git_revision=None, git_dirty_status=None, git_remote=None,
                  git_metadata_reason='Installed distribution has no identifiable source checkout.')
    try:
        direct = distribution('BraTS-evaluation').read_text('direct_url.json')
        if direct:
            info = json.loads(direct)
            result['git_revision'] = info.get('vcs_info', {}).get('commit_id')
            if result['git_revision']:
                result['git_remote'] = info.get('url')
                result['git_metadata_reason'] = 'VCS revision recorded by pip; working-tree status unavailable.'
        for root in Path(package.__file__).resolve().parents:
            if (root / '.git').exists():
                def git(*args):
                    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
                remote = git('remote', 'get-url', 'origin')
                if 'brats/brats_evaluation' in remote.lower():
                    result.update(git_revision=git('rev-parse', 'HEAD'),
                                  git_dirty_status=bool(git('status', '--porcelain')), git_remote=remote,
                                  git_metadata_reason='Observed source checkout.')
                break
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError, PackageNotFoundError):
        pass
    return result


def inspect_gli_configuration(path):
    """Inspect tagged YAML nodes without executing or replacing upstream constructors."""
    def mapping(node):
        if not isinstance(node, yaml.MappingNode):
            raise ValueError('Expected GLI configuration mapping.')
        return {key.value: value for key, value in node.value}
    document = mapping(yaml.compose(path.read_text()))
    if document['expected_input'].value != 'SEMANTIC':
        raise ValueError('GLI evaluator must accept semantic label maps.')
    groups = mapping(mapping(document['segmentation_class_groups'])['groups'])
    expected = dict(netc=[1], snfh=[2], et=[3], rc=[4], wt=[1,2,3], tc=[1,3])
    observed = {name: sorted(int(v.value) for v in mapping(group)['value_labels'].value)
                for name, group in groups.items()}
    if observed != expected:
        raise ValueError(f'Bundled gli label memberships do not match the dataset: {observed}')
    return {'expected_input': 'SEMANTIC', 'label_groups': observed}


def serialized_value(value):
    """Encode upstream nonfinite edge cases explicitly, without replacing scores.

    NaN/Infinity strings are lossless representations of upstream results, not
    locally defined penalty scores or our custom presence-aware conventions.
    """
    if isinstance(value, Real):
        value = float(value) if not isinstance(value, int) else value
        if not math.isfinite(value):
            return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
        return value
    if isinstance(value, dict):
        return {key: serialized_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialized_value(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    raise RuntimeError(f'Incompatible official evaluator value type: {type(value).__name__}')


class OfficialEvaluator:
    def __init__(self):
        try:
            package = importlib.import_module('brats_evaluation')
            panoptica = importlib.import_module('panoptica')
            package_version = version('BraTS-evaluation')
            panoptica_version = version('panoptica')
        except Exception as error:
            raise RuntimeError(f'Official BraTS evaluation package unavailable: install evaluation/requirements_brats2024.txt ({error})') from error
        try:
            config = Path(package.config_path('gli'))
            if not config.is_file():
                raise FileNotFoundError(config)
        except Exception as error:
            raise RuntimeError(f'Official bundled gli configuration unavailable: {error}') from error
        try:
            semantics = inspect_gli_configuration(config)
            self.evaluator = panoptica.Panoptica_Evaluator.load_from_config(str(config))
            self.evaluate_single_exam = package.evaluate_single_exam
            if not callable(self.evaluate_single_exam):
                raise TypeError('evaluate_single_exam is not callable')
        except Exception as error:
            raise RuntimeError(f'Official gli evaluator initialization failed: {error}') from error
        dependencies = {}
        for name in ('numpy', 'nibabel', 'SimpleITK', 'scipy', 'pandas', 'connected-components-3d', 'scikit-image', 'ruamel.yaml'):
            try:
                dependencies[name] = version(name)
            except PackageNotFoundError:
                dependencies[name] = None
        self.metadata = dict(evaluator_package='BraTS-evaluation', evaluator_version=package_version,
                             package_version=package_version, evaluator_configuration='gli',
                             source_repository=REPOSITORY, evaluator_git_revision=source_git_metadata(package)['git_revision'],
                             panoptica_version=panoptica_version, dependency_versions=dependencies,
                             bundled_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                             evaluation_terminology=TERMINOLOGY,
                             is_official_hidden_challenge_scoring=False,
                             nonfinite_serialization='Upstream NaN/Infinity/-Infinity encoded as literal strings; no score substitution')
        self.audit = dict(evaluator_repository=REPOSITORY, evaluator_package='BraTS-evaluation',
                          evaluator_package_version=package_version, evaluator_configuration='gli',
                          evaluator_configuration_path=str(config.resolve()),
                          evaluator_configuration_sha256=self.metadata['bundled_config_sha256'],
                          Panoptica_version=panoptica_version, Python_version=platform.python_version(),
                          relevant_dependency_versions=dependencies,
                          evaluation_API_used='Panoptica_Evaluator.load_from_config(str(config_path("gli"))); brats_evaluation.evaluate_single_exam(prediction_filepath=..., reference_filepath=..., subject_identifier=..., evaluator=...)',
                          configuration_semantics=semantics, **source_git_metadata(package), audit_passed=True)
        self.metadata.update(evaluator_configuration_path=str(config.resolve()),
                             evaluator_configuration_sha256=self.metadata['bundled_config_sha256'])
        self.raw_results = []

    def evaluate(self, prediction, reference, workspace):
        prediction = Path(prediction).resolve()
        case_id = prediction.name.removesuffix('.nii.gz')
        try:
            result = self.evaluate_single_exam(
                prediction_filepath=str(prediction), reference_filepath=str(Path(reference).resolve()),
                subject_identifier=case_id, evaluator=self.evaluator)
        except Exception as error:
            raise RuntimeError(f'Official evaluator failed (no fallback): {error}') from error
        # The official wrapper catches exceptions and returns an error dictionary.
        if isinstance(result, dict) and 'error' in result:
            raise RuntimeError(f"Official evaluator failed (no fallback): {result['error']}")
        expected = {r.lower() for r in REGIONS} | {'subject_name'}
        if not isinstance(result, dict) or set(result) != expected or result['subject_name'] != case_id:
            raise RuntimeError('Incompatible official evaluator result schema or subject identifier.')
        required = {'tp', 'fp', 'fn', 'sq_dsc', 'sq_hd95', 'global_bin_dsc', 'global_bin_hd95'}
        rows = []
        for region in REGIONS:
            metrics = result[region.lower()]
            if not isinstance(metrics, dict) or not required <= metrics.keys():
                raise RuntimeError(f'Incompatible official evaluator metric schema for {region}.')
            if any(not isinstance(metrics[key], Real) for key in required):
                raise RuntimeError(f'Incompatible official evaluator numeric metrics for {region}.')
            encoded = serialized_value(metrics)
            # Nested upstream instance details are retained as JSON-valued CSV cells.
            rows.append(dict(Labels=region, **{key: json.dumps(value, allow_nan=False, sort_keys=True)
                                              if isinstance(value, (list, dict)) else value
                                              for key, value in encoded.items()}))
        self.raw_results.append(serialized_value(result))
        return rows
