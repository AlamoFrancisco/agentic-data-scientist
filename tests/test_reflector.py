"""
Tests for agents/reflector.py

Covers: reflect (classification + regression paths), should_replan,
apply_replan_strategy (overfitting, low-F1, baseline, imbalance branches).
"""
from agents.reflector import reflect, should_replan, apply_replan_strategy


# ── helpers ───────────────────────────────────────────────────────────────────

def cls_profile(imb=1.0):
    return {"is_classification": True, "imbalance_ratio": imb, "missing_pct": {}}


def reg_profile():
    return {"is_classification": False, "imbalance_ratio": None, "missing_pct": {}}


def cls_eval(bal_acc=0.85, f1=0.80, model="RandomForest"):
    return {"model": model, "balanced_accuracy": bal_acc, "f1_macro": f1}


def cls_all(bal_acc=0.85, f1=0.80):
    return [
        {"model": "RandomForest",      "balanced_accuracy": bal_acc, "f1_macro": f1},
        {"model": "DummyMostFrequent", "balanced_accuracy": 0.50,    "f1_macro": 0.25},
    ]


def reg_eval(r2=0.8, model="RandomForestRegressor"):
    return {"model": model, "r2": r2}


def reg_all(r2=0.8):
    return [
        {"model": "RandomForestRegressor", "r2": r2},
        {"model": "DummyMean",             "r2": 0.0},
    ]


# ── reflect – classification ──────────────────────────────────────────────────

def test_classification_good_performance_is_ok():
    result = reflect(cls_profile(), cls_eval(0.85, 0.80), cls_all(0.85, 0.80))
    assert result["status"] == "ok"
    assert result["best_model"] == "RandomForest"
    assert result["issues"] == []


def test_classification_low_f1_adds_issue():
    result = reflect(cls_profile(), cls_eval(0.85, 0.40), cls_all(0.85, 0.40))
    assert result["status"] == "needs_attention"
    assert any("F1" in i or "f1" in i.lower() for i in result["issues"])


def test_classification_overfitting_detected():
    all_m = [
        {"model": "RandomForest",      "balanced_accuracy": 0.95, "f1_macro": 0.55},
        {"model": "DummyMostFrequent", "balanced_accuracy": 0.40, "f1_macro": 0.20},
    ]
    result = reflect(cls_profile(), cls_eval(0.95, 0.55), all_m)
    assert any("overfitting" in i.lower() for i in result["issues"])


def test_classification_imbalance_adds_suggestion():
    result = reflect(cls_profile(imb=5.0), cls_eval(), cls_all())
    assert any("imbalance" in s.lower() or "class_weight" in s.lower() for s in result["suggestions"])


def test_classification_low_diversity_adds_issue():
    # All models have nearly identical f1 scores
    all_m = [
        {"model": "RandomForest",      "balanced_accuracy": 0.80, "f1_macro": 0.79},
        {"model": "LogisticRegression","balanced_accuracy": 0.80, "f1_macro": 0.79},
        {"model": "DummyMostFrequent", "balanced_accuracy": 0.80, "f1_macro": 0.79},
    ]
    result = reflect(cls_profile(), cls_eval(0.80, 0.79), all_m)
    assert any("diversity" in i.lower() or "similar" in i.lower() for i in result["issues"])


def test_classification_weak_baseline_adds_issue():
    # Best model barely beats dummy
    all_m = [
        {"model": "RandomForest",      "balanced_accuracy": 0.52, "f1_macro": 0.50},
        {"model": "DummyMostFrequent", "balanced_accuracy": 0.50, "f1_macro": 0.25},
    ]
    result = reflect(cls_profile(), cls_eval(0.52, 0.50), all_m)
    assert any("baseline" in i.lower() for i in result["issues"])


def test_classification_near_perfect_multiple_models_flags_suspicion():
    all_m = [
        {"model": "LogisticRegression", "balanced_accuracy": 1.00, "f1_macro": 1.00},
        {"model": "RandomForest",       "balanced_accuracy": 1.00, "f1_macro": 1.00},
        {"model": "DummyMostFrequent",  "balanced_accuracy": 0.33, "f1_macro": 0.20},
    ]
    result = reflect(
        cls_profile(),
        cls_eval(1.00, 1.00, model="LogisticRegression"),
        all_m,
    )
    assert result["status"] == "needs_attention"
    assert any("suspicious" in i.lower() for i in result["issues"])
    assert any("leakage" in s.lower() or "target prox" in s.lower() for s in result["suggestions"])


def test_classification_single_near_perfect_model_does_not_flag_suspicion():
    all_m = [
        {"model": "RandomForest",      "balanced_accuracy": 1.00, "f1_macro": 1.00},
        {"model": "LogisticRegression","balanced_accuracy": 0.91, "f1_macro": 0.90},
        {"model": "DummyMostFrequent", "balanced_accuracy": 0.50, "f1_macro": 0.25},
    ]
    result = reflect(
        cls_profile(),
        cls_eval(1.00, 1.00),
        all_m,
    )
    assert not any("suspicious" in i.lower() for i in result["issues"])


# ── reflect – regression ─────────────────────────────────────────────────────

def test_regression_good_performance_is_ok():
    result = reflect(reg_profile(), reg_eval(0.8), reg_all(0.8))
    assert result["status"] == "ok"
    assert result["best_model"] == "RandomForestRegressor"


def test_regression_low_r2_adds_issue():
    result = reflect(reg_profile(), reg_eval(0.05), reg_all(0.05))
    assert result["status"] == "needs_attention"
    assert any("R²" in i or "r2" in i.lower() for i in result["issues"])


def test_regression_weak_baseline_adds_issue():
    # Best r2=0.04, dummy r2=0.03 → improvement < 0.05
    all_m = [
        {"model": "RF",        "r2": 0.04},
        {"model": "DummyMean", "r2": 0.03},
    ]
    result = reflect(reg_profile(), reg_eval(0.04, model="RF"), all_m)
    assert any("baseline" in i.lower() for i in result["issues"])


# ── should_replan ─────────────────────────────────────────────────────────────

def test_should_replan_true_when_recommended():
    assert should_replan({"replan_recommended": True, "issues": [], "suggestions": [], "status": "ok"}) is True


def test_should_replan_true_multiple_issues():
    assert should_replan({"replan_recommended": False, "issues": ["a", "b"], "suggestions": [], "status": "ok"}) is True


def test_should_replan_true_needs_attention_with_suggestions():
    r = {"replan_recommended": False, "issues": ["one"], "suggestions": ["try X"], "status": "needs_attention"}
    assert should_replan(r) is True


def test_should_replan_false_when_clean():
    assert should_replan({"replan_recommended": False, "issues": [], "suggestions": [], "status": "ok"}) is False


# ── apply_replan_strategy ─────────────────────────────────────────────────────

BASE_PLAN = ["profile_dataset", "build_preprocessor", "train_models", "evaluate"]


def test_apply_replan_always_adds_attempt():
    new_plan, new_profile = apply_replan_strategy(BASE_PLAN, {"notes": []}, {"issues": [], "suggestions": []})
    assert "replan_attempt" in new_plan
    assert any("Replan" in n for n in new_profile["notes"])


def test_apply_replan_overfitting_adds_regularization():
    reflection = {"issues": ["High balanced accuracy but low F1 macro suggests overfitting."], "suggestions": []}
    new_plan, _ = apply_replan_strategy(BASE_PLAN, {"notes": []}, reflection)
    assert "apply_regularization" in new_plan


def test_apply_replan_low_f1_adds_ensemble():
    reflection = {"issues": ["Macro F1 score is modest (<0.60)."], "suggestions": []}
    new_plan, _ = apply_replan_strategy(BASE_PLAN, {"notes": []}, reflection)
    assert "try_ensemble_methods" in new_plan


def test_apply_replan_baseline_issue_adds_feature_engineering():
    reflection = {"issues": ["Best model only 0.02 better than baseline."], "suggestions": []}
    new_plan, _ = apply_replan_strategy(BASE_PLAN, {"notes": []}, reflection)
    assert "apply_feature_engineering" in new_plan


def test_apply_replan_does_not_modify_original_plan():
    original = list(BASE_PLAN)
    reflection = {"issues": ["Macro F1 score is modest (<0.60)."], "suggestions": []}
    apply_replan_strategy(BASE_PLAN, {"notes": []}, reflection)
    assert BASE_PLAN == original
