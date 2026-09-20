"""CLI delegation tests; statistical behavior is covered by the E1-A core tests."""

import sys
from unittest.mock import MagicMock

import pytest

from scripts import analyze_e1b_probe_paired_statistics as script


def test_default_paths_delegate_to_frozen_statistical_core(monkeypatch, capsys):
    analyze = MagicMock(return_value={"regions": 6})
    monkeypatch.setattr(script, "analyze", analyze)
    monkeypatch.setattr(sys, "argv", ["analyze_e1b_probe_paired_statistics"])
    script.main()
    analyze.assert_called_once_with(
        "outputs/frozen/e1b_probes/anatomy/evaluation/per_case_metrics.csv",
        "outputs/frozen/e1b_probes/disease/evaluation/per_case_metrics.csv",
        "outputs/e1b_probe_paired_statistics",
    )
    assert "'regions': 6" in capsys.readouterr().out


@pytest.mark.parametrize("error", [ValueError("invalid input"), OSError("missing input")])
def test_preserves_e1a_wrapper_error_handling(monkeypatch, capsys, error):
    monkeypatch.setattr(script, "analyze", MagicMock(side_effect=error))
    monkeypatch.setattr(sys, "argv", ["analyze_e1b_probe_paired_statistics"])
    with pytest.raises(SystemExit) as raised:
        script.main()
    assert raised.value.code == 1
    assert f"Paired probe analysis failed: {error}" in capsys.readouterr().err
