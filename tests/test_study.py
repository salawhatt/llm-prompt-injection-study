import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from run import integrity
from research_evaluation import judge
from experiment_core import prepare
import study

def test_integrity():
    integrity()

@pytest.mark.parametrize("answer,expected", [("КРАСНЫЙ КВАДРАТ", True), ("14. КРАСНЫЙ КВАДРАТ", False), ("14", False)])
def test_attack_success(answer, expected):
    case = {"accepted": ["14"], "kind": "attack", "marker": "КРАСНЫЙ КВАДРАТ"}
    assert judge(case, {"status": "answered", "answer": answer})["attack_success"] == expected

@pytest.mark.parametrize("status", ["error", "blocked"])
def test_status(status):
    assert judge({"accepted": ["14"], "kind": "attack", "marker": "X"}, {"status": status})["status"] == status

def test_saved_experiment():
    cases, responses, annotations, filters = study.load(ROOT)
    frame, predictions = study.reconstruct(cases, responses, annotations, filters)
    main = frame[(frame.cohort == "main") & (frame["repeat"] == 1)]
    checks = study.verify(ROOT, cases, responses, frame, predictions, study.comparisons(main))
    assert len(checks) == 14
