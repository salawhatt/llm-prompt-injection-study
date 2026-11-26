"""Offline controls for the published hard-challenge snapshot."""
import builtins
import csv
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("public_challenge_wrapper", ROOT / "analyze_saved.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_original_archive_and_real_observation_counts():
    result = wrapper.verify_snapshot()
    assert result["unique_observations"] == 1366
    assert result["observations_by_model"] == {"sonnet55": 683, "astra": 683}
    assert result["distinct_situations"] == 72
    assert result["repeated_trials_complete"] is False


def test_copy_inputs_preserves_originals_and_omits_runtime_ledger(tmp_path):
    for original in wrapper.input_paths():
        target = tmp_path / original.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    before = {p: wrapper.sha256(p) for p in wrapper.input_paths(tmp_path)}
    output = wrapper.copy_inputs(tmp_path)
    assert output == tmp_path / "output/reproduced"
    assert not (output / "budget.sqlite").exists()
    assert not (output / "budget_authorization.json").exists()
    assert len(list(output.rglob("*.*"))) == len(before)
    for original, digest in before.items():
        assert wrapper.sha256(original) == digest
        assert wrapper.sha256(output / original.relative_to(tmp_path / wrapper.RESULT_PATH)) == digest
    wrapper.copy_inputs(tmp_path)
    assert all(wrapper.sha256(p) == digest for p, digest in before.items())


def test_verifier_and_analytics_do_not_import_api_or_training(monkeypatch):
    original_import = builtins.__import__
    forbidden = {"client", "httpx", "openai", "requests", "research_defense", "research_data", "joblib", "sklearn", "continuation", "validate_artifacts"}

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in forbidden, "Unexpected runtime import: " + name
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded)
        wrapper.verify_snapshot()
        analysis, post = wrapper.load_analytics()
        assert callable(analysis.reconstruct) and callable(post.run)


def test_frozen_analytics_reproduce_real_asr_without_bootstrap():
    analysis, post = wrapper.load_analytics()
    stage = post.load_stage(ROOT / wrapper.RESULT_PATH)
    first = stage["frame"][stage["frame"]["repeat"].eq(1)]
    summary = post.describe(first)
    expected = {
        ("sonnet55", "A"): 19, ("sonnet55", "B"): 0, ("sonnet55", "D"): 0, ("sonnet55", "E"): 0,
        ("astra", "A"): 3, ("astra", "B"): 1, ("astra", "D"): 0, ("astra", "E"): 0,
    }
    for row in summary[summary.kind.eq("attack")].itertuples():
        assert row.completed == 64
        assert row.successes == expected[row.model_name, row.arm]
        assert row.asr == pytest.approx(row.successes / 64)
    robust = post.load_stage(ROOT / wrapper.RESULT_PATH / "robustness")
    attacks = robust["frame"]
    assert int(attacks[(attacks.model_name == "astra") & (attacks.arm == "B")].attack_success.sum()) == 2
    assert int(attacks[(attacks.model_name == "astra") & (attacks.arm == "D")].attack_success.sum()) == 1
    assert int(attacks[(attacks.model_name == "astra") & (attacks.arm == "E")].attack_success.sum()) == 0
    assert int(attacks.status.eq("missing").sum()) > 0
    supplement = post.load_stage(ROOT / wrapper.RESULT_PATH / "supplement")
    assert int(supplement["frame"].attack_success.sum()) == 0


def test_archive_hash_change_is_rejected_before_analysis(tmp_path):
    path = tmp_path / "changed.txt"
    path.write_text("changed", encoding="utf8")
    manifest = {"study": wrapper.STUDY, "files": {"changed.txt": {"bytes": 7, "sha256": "0" * 64}}}
    (tmp_path / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="Archived file changed"):
        wrapper.verify_snapshot(tmp_path)


def test_archive_path_escape_is_rejected(tmp_path):
    manifest = {"study": wrapper.STUDY, "files": {"../outside.txt": {"bytes": 0, "sha256": "0" * 64}}}
    (tmp_path / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="Unsafe archive manifest path"):
        wrapper.verify_snapshot(tmp_path)
