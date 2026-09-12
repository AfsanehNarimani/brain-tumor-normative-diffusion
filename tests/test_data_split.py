"""Synthetic stratification allocation and summary checks."""
import pytest
from scripts.create_data_split import split_cases, summarize


def metadata():
    return {f'case-{i:03}': (1, 1, 1, 1) if i < 96 else (0, 1, 0, 1) for i in range(100)}


def test_deterministic_coverage_and_rare_stratum():
    data = metadata()
    train, val = split_cases(data)
    assert (train, val) == split_cases(dict(reversed(list(data.items()))))
    assert (train, val) != split_cases(data, seed=43)
    assert train == sorted(train) and val == sorted(val)
    assert len(train) == 80 and len(val) == 20
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(data)
    assert sum(data[c] == (0, 1, 0, 1) for c in train) == 3
    assert sum(data[c] == (0, 1, 0, 1) for c in val) == 1
    summary = summarize(data, train, val, 0.8, 42)
    assert summary['full']['combinations'] == {'1111': 96, '0101': 4}
    assert summary['train']['tissues']['NETC'] == {'count': 77, 'percentage': 96.25}
    assert summary['validation']['tissues']['RC'] == {'count': 20, 'percentage': 100.0}
    assert summary['case_presence']['case-099'] == [0, 1, 0, 1]


@pytest.mark.parametrize('ratio', [0.1, 0.5, 0.8, 0.99])
def test_singletons_and_small_strata(ratio):
    data = {'a': (1, 0, 0, 0), 'b': (1, 0, 0, 0), 'c': (0, 0, 0, 1)}
    train, val = split_cases(data, ratio)
    assert len(train) + len(val) == 3
    assert train and val and not set(train) & set(val)
    assert ('a' in train) != ('b' in train)


def test_rare_bounds_relaxed_if_infeasible():
    data = {'a': (1, 0, 0, 0), 'b': (1, 0, 0, 0), 'c': (0, 0, 0, 1), 'd': (0, 0, 0, 1)}
    train, val = split_cases(data, 0.9)
    assert len(train) == 3 and len(val) == 1


@pytest.mark.parametrize('ratio', [0, 1, float('nan')])
def test_invalid_ratio(ratio):
    with pytest.raises(ValueError):
        split_cases(metadata(), ratio)
