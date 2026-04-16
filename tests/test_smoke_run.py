import os
import sys
import json

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agentic_data_scientist as ads
from agentic_data_scientist import AgenticDataScientist

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def test_smoke_run_creates_outputs(tmp_path):
    """Full pipeline runs without error and produces expected output files."""
    agent = AgenticDataScientist(
        memory_path=str(tmp_path / "memory.json"),
        verbose=False,
    )
    out_dir = agent.run(
        data_path=os.path.join(DATA_DIR, "penguins.csv"),
        target="species",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )
    assert os.path.isdir(out_dir), "Output directory was not created"
    for fname in ("metrics.json", "plan.json", "reflection.json", "eda_summary.json", "report.md"):
        assert os.path.exists(os.path.join(out_dir, fname)), f"Missing output: {fname}"


def test_smoke_run_metrics_structure(tmp_path):
    """Metrics file contains expected keys."""
    agent = AgenticDataScientist(
        memory_path=str(tmp_path / "memory.json"),
        verbose=False,
    )
    out_dir = agent.run(
        data_path=os.path.join(DATA_DIR, "penguins.csv"),
        target="species",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )
    with open(os.path.join(out_dir, "metrics.json")) as f:
        metrics = json.load(f)
    assert "best_metrics" in metrics
    assert "all_metrics" in metrics
    assert "cross_validation" in metrics
    assert "model" in metrics["best_metrics"]
    assert "enabled" in metrics["cross_validation"]


def test_smoke_run_auto_target(tmp_path):
    """Auto target detection runs without error."""
    agent = AgenticDataScientist(
        memory_path=str(tmp_path / "memory.json"),
        verbose=False,
    )
    out_dir = agent.run(
        data_path=os.path.join(DATA_DIR, "penguins.csv"),
        target="auto",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )
    assert os.path.isdir(out_dir)


def test_auto_target_ignores_memory_targets_that_were_only_inferred(monkeypatch, tmp_path):
    data_path = tmp_path / "memory_target.csv"
    pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "manual_target": [0, 1, 0, 1],
            "inferred_target": [1, 1, 0, 0],
        }
    ).to_csv(data_path, index=False)

    memory_path = tmp_path / "memory.json"
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": {
                    "fp_inferred": {
                        "dataset": data_path.name,
                        "target": "manual_target",
                        "target_source": "inferred",
                        "target_origin": "inferred",
                        "shape": {"rows": 4, "cols": 3},
                        "verdict_label": "Reliable result",
                        "best_model": "RandomForest",
                    }
                },
                "notes": [],
                "failed_targets": {},
            },
            f,
            indent=2,
        )

    agent = AgenticDataScientist(memory_path=str(memory_path), verbose=False)

    monkeypatch.setattr(ads, "infer_target_column", lambda df, return_scores=False: ("inferred_target", {"inferred_target": 5.0, "manual_target": 4.0}) if return_scores else "inferred_target")
    monkeypatch.setattr(
        ads,
        "profile_dataset",
        lambda df, target, **kwargs: {
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "is_classification": True,
            "notes": [],
        },
    )
    monkeypatch.setattr(
        ads,
        "create_plan",
        lambda profile, memory_hint=None: ["profile_dataset", "build_preprocessor", "select_models", "train_models", "evaluate", "write_report"],
    )
    monkeypatch.setattr(ads, "build_preprocessor", lambda profile: object())
    monkeypatch.setattr(ads, "select_models", lambda profile, seed, preferred_model=None: [("DummyMostFrequent", object()), ("LogisticRegression", object())])
    monkeypatch.setattr(
        ads,
        "train_models",
        lambda **kwargs: {
            "best": {
                "name": "LogisticRegression",
                "metrics": {"model": "LogisticRegression", "balanced_accuracy": 0.80, "f1_macro": 0.79},
            },
            "all_metrics": [
                {"model": "DummyMostFrequent", "balanced_accuracy": 0.50, "f1_macro": 0.33},
                {"model": "LogisticRegression", "balanced_accuracy": 0.80, "f1_macro": 0.79},
            ],
            "results": [],
            "training_warnings": [],
        },
    )
    monkeypatch.setattr(
        ads,
        "evaluate_best",
        lambda training_payload, output_dir, is_classification=True: {
            "best_metrics": {"model": "LogisticRegression", "balanced_accuracy": 0.80, "f1_macro": 0.79},
            "all_metrics": training_payload["all_metrics"],
        },
    )
    monkeypatch.setattr(
        ads,
        "reflect",
        lambda **kwargs: {
            "status": "ok",
            "issues": [],
            "suggestions": [],
            "replan_recommended": False,
            "review_required": False,
            "training_warnings": [],
        },
    )
    monkeypatch.setattr(
        ads,
        "derive_run_verdict",
        lambda dataset_profile, eval_payload, reflection: {"label": "Reliable result", "detail": "ok"},
    )
    monkeypatch.setattr(ads, "should_replan", lambda reflection: False)
    monkeypatch.setattr(ads, "write_markdown_report", lambda **kwargs: None)

    out_dir = agent.run(
        data_path=str(data_path),
        target="auto",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )

    assert os.path.isdir(out_dir)
    assert agent.ctx.target == "inferred_target"


def test_smoke_run_memory_persists(tmp_path):
    """Second run on same dataset writes a memory record."""
    memory_path = str(tmp_path / "memory.json")
    data_path = os.path.join(DATA_DIR, "penguins.csv")

    agent = AgenticDataScientist(memory_path=memory_path, verbose=False)
    agent.run(data_path=data_path, target="species",
              output_root=str(tmp_path / "run1"), seed=42, max_replans=0)

    with open(memory_path) as f:
        mem = json.load(f)
    assert len(mem.get("datasets", {})) > 0, "Memory was not written after first run"
    assert any("verdict_label" in record for record in mem.get("datasets", {}).values())


def test_smoke_run_regression(tmp_path):
    """Pipeline handles a regression dataset end-to-end."""
    agent = AgenticDataScientist(
        memory_path=str(tmp_path / "memory.json"),
        verbose=False,
    )
    out_dir = agent.run(
        data_path=os.path.join(DATA_DIR, "WineQuality.csv"),
        target="quality",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )
    with open(os.path.join(out_dir, "eda_summary.json")) as f:
        profile = json.load(f)
    assert profile["is_classification"] is False


def test_smoke_run_verbose_output_uses_compact_summary(tmp_path, capsys):
    agent = AgenticDataScientist(
        memory_path=str(tmp_path / "memory.json"),
        verbose=True,
    )
    agent.run(
        data_path=os.path.join(DATA_DIR, "penguins.csv"),
        target="species",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )
    output = capsys.readouterr().out
    assert "Decision summary:" in output
    assert "Final summary:" in output
    assert "Verdict:" in output
    assert "Plan includes" not in output


def test_replan_diff_logging_is_compact(tmp_path, capsys):
    agent = AgenticDataScientist(
        memory_path=str(tmp_path / "memory.json"),
        verbose=True,
    )
    agent._log_replan_diff(
        ["profile_dataset", "train_models"],
        ["profile_dataset", "train_models", "use_ensemble_models", "replan_attempt"],
        {"issues": ["Best model only 0.02 better than baseline."]},
    )
    output = capsys.readouterr().out
    assert "Replan changes:" in output
    assert "added" in output
    assert "Replan reason: Best model only 0.02 better than baseline." in output


def test_auto_target_fallback_tries_next_candidate_and_marks_failed(monkeypatch, tmp_path):
    data_path = tmp_path / "auto_targets.csv"
    pd.DataFrame(
        {
            "weak_target": [0, 1, 0, 1],
            "strong_target": [1, 1, 0, 0],
            "feature": [10, 11, 12, 13],
        }
    ).to_csv(data_path, index=False)

    memory_path = tmp_path / "memory.json"
    agent = AgenticDataScientist(memory_path=str(memory_path), verbose=False)

    def fake_infer_target_column(df, return_scores=False):
        scores = {"weak_target": 5.0, "strong_target": 4.0}
        if return_scores:
            return "weak_target", scores
        return "weak_target"

    def fake_profile_dataset(df, target, **kwargs):
        return {
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "is_classification": True,
            "notes": [],
        }

    def fake_create_plan(profile, memory_hint=None):
        return ["profile_dataset", "build_preprocessor", "select_models", "train_models", "evaluate", "write_report"]

    def fake_select_models(profile, seed, preferred_model=None):
        return [("DummyMostFrequent", object()), ("LogisticRegression", object())]

    def fake_train_models(**kwargs):
        return {"training_warnings": []}

    def fake_evaluate_best(training_payload, output_dir, is_classification=True):
        if agent.ctx.target == "weak_target":
            return {
                "best_metrics": {
                    "model": "LogisticRegression",
                    "balanced_accuracy": 0.21,
                    "f1_macro": 0.20,
                },
                "all_metrics": [
                    {
                        "model": "DummyMostFrequent",
                        "balanced_accuracy": 0.20,
                        "f1_macro": 0.20,
                    },
                    {
                        "model": "LogisticRegression",
                        "balanced_accuracy": 0.21,
                        "f1_macro": 0.20,
                    },
                ],
            }
        return {
            "best_metrics": {
                "model": "LogisticRegression",
                "balanced_accuracy": 0.82,
                "f1_macro": 0.81,
            },
            "all_metrics": [
                {
                    "model": "DummyMostFrequent",
                    "balanced_accuracy": 0.50,
                    "f1_macro": 0.33,
                },
                {
                    "model": "LogisticRegression",
                    "balanced_accuracy": 0.82,
                    "f1_macro": 0.81,
                },
            ],
        }

    def fake_reflect(**kwargs):
        return {
            "status": "ok",
            "issues": [],
            "suggestions": [],
            "replan_recommended": False,
            "review_required": False,
            "training_warnings": [],
        }

    def fake_derive_run_verdict(dataset_profile, eval_payload, reflection):
        if agent.ctx.target == "weak_target":
            return {"label": "Use with caution", "detail": "Weak target signal."}
        return {"label": "Reliable result", "detail": "Strong enough target signal."}

    monkeypatch.setattr(ads, "infer_target_column", fake_infer_target_column)
    monkeypatch.setattr(ads, "profile_dataset", fake_profile_dataset)
    monkeypatch.setattr(ads, "create_plan", fake_create_plan)
    monkeypatch.setattr(ads, "build_preprocessor", lambda profile: object())
    monkeypatch.setattr(ads, "select_models", fake_select_models)
    monkeypatch.setattr(ads, "train_models", fake_train_models)
    monkeypatch.setattr(ads, "evaluate_best", fake_evaluate_best)
    monkeypatch.setattr(ads, "reflect", fake_reflect)
    monkeypatch.setattr(ads, "derive_run_verdict", fake_derive_run_verdict)
    monkeypatch.setattr(ads, "should_replan", lambda reflection: False)
    monkeypatch.setattr(ads, "write_markdown_report", lambda **kwargs: None)

    out_dir = agent.run(
        data_path=str(data_path),
        target="auto",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )

    assert os.path.isdir(out_dir)
    assert agent.ctx.target == "strong_target"

    with open(memory_path) as f:
        memory = json.load(f)
    assert memory["failed_targets"][data_path.name] == ["weak_target"]
    stored_targets = {record["target"] for record in memory.get("datasets", {}).values()}
    assert "strong_target" in stored_targets


def test_run_skips_tuning_when_signal_is_too_weak(monkeypatch, tmp_path):
    data_path = tmp_path / "weak_signal.csv"
    pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    ).to_csv(data_path, index=False)

    agent = AgenticDataScientist(
        memory_path=str(tmp_path / "memory.json"),
        verbose=False,
    )
    tune_called = {"value": False}

    monkeypatch.setattr(
        ads,
        "profile_dataset",
        lambda df, target, **kwargs: {
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "is_classification": True,
            "notes": [],
        },
    )
    monkeypatch.setattr(
        ads,
        "create_plan",
        lambda profile, memory_hint=None: [
            "profile_dataset",
            "build_preprocessor",
            "select_models",
            "train_models",
            "tune_hyperparameters",
            "evaluate",
            "write_report",
        ],
    )
    monkeypatch.setattr(ads, "build_preprocessor", lambda profile: object())
    monkeypatch.setattr(
        ads,
        "select_models",
        lambda profile, seed, preferred_model=None: [
            ("DummyMostFrequent", object()),
            ("LogisticRegression", object()),
        ],
    )
    monkeypatch.setattr(
        ads,
        "train_models",
        lambda **kwargs: {
            "best": {
                "name": "LogisticRegression",
                "metrics": {
                    "model": "LogisticRegression",
                    "balanced_accuracy": 0.21,
                    "f1_macro": 0.20,
                },
            },
            "all_metrics": [
                {
                    "model": "DummyMostFrequent",
                    "balanced_accuracy": 0.20,
                    "f1_macro": 0.20,
                },
                {
                    "model": "LogisticRegression",
                    "balanced_accuracy": 0.21,
                    "f1_macro": 0.20,
                },
            ],
            "results": [],
            "training_warnings": [],
        },
    )

    def fake_tune_best_model(training_payload, seed=42, is_classification=True):
        tune_called["value"] = True
        return training_payload

    monkeypatch.setattr(ads, "tune_best_model", fake_tune_best_model)
    monkeypatch.setattr(
        ads,
        "evaluate_best",
        lambda training_payload, output_dir, is_classification=True: {
            "best_metrics": {
                "model": "LogisticRegression",
                "balanced_accuracy": 0.21,
                "f1_macro": 0.20,
            },
            "all_metrics": training_payload["all_metrics"],
        },
    )
    monkeypatch.setattr(
        ads,
        "reflect",
        lambda **kwargs: {
            "status": "needs_attention",
            "issues": ["Weak improvement over baseline."],
            "suggestions": [],
            "replan_recommended": False,
            "review_required": False,
            "training_warnings": [],
        },
    )
    monkeypatch.setattr(
        ads,
        "derive_run_verdict",
        lambda dataset_profile, eval_payload, reflection: {
            "label": "Use with caution",
            "detail": "Weak signal.",
        },
    )
    monkeypatch.setattr(ads, "should_replan", lambda reflection: False)
    monkeypatch.setattr(ads, "write_markdown_report", lambda **kwargs: None)

    out_dir = agent.run(
        data_path=str(data_path),
        target="target",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )

    assert os.path.isdir(out_dir)
    assert tune_called["value"] is False

    with open(os.path.join(out_dir, "eda_summary.json")) as f:
        profile = json.load(f)
    assert any("Skipped hyperparameter tuning" in note for note in profile.get("notes", []))


def test_run_skips_tuning_when_result_is_already_near_perfect(monkeypatch, tmp_path):
    data_path = tmp_path / "near_perfect.csv"
    pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    ).to_csv(data_path, index=False)

    agent = AgenticDataScientist(
        memory_path=str(tmp_path / "memory.json"),
        verbose=False,
    )
    tune_called = {"value": False}

    monkeypatch.setattr(
        ads,
        "profile_dataset",
        lambda df, target, **kwargs: {
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "is_classification": True,
            "notes": [],
        },
    )
    monkeypatch.setattr(
        ads,
        "create_plan",
        lambda profile, memory_hint=None: [
            "profile_dataset",
            "build_preprocessor",
            "select_models",
            "train_models",
            "tune_hyperparameters",
            "evaluate",
            "write_report",
        ],
    )
    monkeypatch.setattr(ads, "build_preprocessor", lambda profile: object())
    monkeypatch.setattr(
        ads,
        "select_models",
        lambda profile, seed, preferred_model=None: [
            ("DummyMostFrequent", object()),
            ("LogisticRegression", object()),
        ],
    )
    monkeypatch.setattr(
        ads,
        "train_models",
        lambda **kwargs: {
            "best": {
                "name": "LogisticRegression",
                "metrics": {
                    "model": "LogisticRegression",
                    "balanced_accuracy": 1.0,
                    "f1_macro": 1.0,
                },
            },
            "all_metrics": [
                {
                    "model": "DummyMostFrequent",
                    "balanced_accuracy": 0.50,
                    "f1_macro": 0.33,
                },
                {
                    "model": "LogisticRegression",
                    "balanced_accuracy": 1.0,
                    "f1_macro": 1.0,
                },
            ],
            "results": [],
            "training_warnings": [],
        },
    )

    def fake_tune_best_model(training_payload, seed=42, is_classification=True):
        tune_called["value"] = True
        return training_payload

    monkeypatch.setattr(ads, "tune_best_model", fake_tune_best_model)
    monkeypatch.setattr(
        ads,
        "evaluate_best",
        lambda training_payload, output_dir, is_classification=True: {
            "best_metrics": {
                "model": "LogisticRegression",
                "balanced_accuracy": 1.0,
                "f1_macro": 1.0,
            },
            "all_metrics": training_payload["all_metrics"],
        },
    )
    monkeypatch.setattr(
        ads,
        "reflect",
        lambda **kwargs: {
            "status": "needs_attention",
            "issues": ["Near-perfect performance is suspicious."],
            "suggestions": [],
            "replan_recommended": False,
            "review_required": True,
            "training_warnings": [],
        },
    )
    monkeypatch.setattr(
        ads,
        "derive_run_verdict",
        lambda dataset_profile, eval_payload, reflection: {
            "label": "Use with caution",
            "detail": "Near-perfect performance should be reviewed.",
        },
    )
    monkeypatch.setattr(ads, "should_replan", lambda reflection: False)
    monkeypatch.setattr(ads, "write_markdown_report", lambda **kwargs: None)

    out_dir = agent.run(
        data_path=str(data_path),
        target="target",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )

    assert os.path.isdir(out_dir)
    assert tune_called["value"] is False

    with open(os.path.join(out_dir, "eda_summary.json")) as f:
        profile = json.load(f)
    assert any("near-perfect" in note.lower() for note in profile.get("notes", []))


def test_run_skips_tuning_when_training_is_numerically_unstable(monkeypatch, tmp_path):
    data_path = tmp_path / "unstable_training.csv"
    pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    ).to_csv(data_path, index=False)

    agent = AgenticDataScientist(
        memory_path=str(tmp_path / "memory.json"),
        verbose=False,
    )
    tune_called = {"value": False}

    monkeypatch.setattr(
        ads,
        "profile_dataset",
        lambda df, target, **kwargs: {
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "is_classification": True,
            "notes": [],
        },
    )
    monkeypatch.setattr(
        ads,
        "create_plan",
        lambda profile, memory_hint=None: [
            "profile_dataset",
            "build_preprocessor",
            "select_models",
            "train_models",
            "tune_hyperparameters",
            "evaluate",
            "write_report",
        ],
    )
    monkeypatch.setattr(ads, "build_preprocessor", lambda profile: object())
    monkeypatch.setattr(
        ads,
        "select_models",
        lambda profile, seed, preferred_model=None: [
            ("DummyMostFrequent", object()),
            ("LogisticRegression", object()),
        ],
    )
    monkeypatch.setattr(
        ads,
        "train_models",
        lambda **kwargs: {
            "best": {
                "name": "LogisticRegression",
                "metrics": {
                    "model": "LogisticRegression",
                    "balanced_accuracy": 0.85,
                    "f1_macro": 0.84,
                },
            },
            "all_metrics": [
                {
                    "model": "DummyMostFrequent",
                    "balanced_accuracy": 0.50,
                    "f1_macro": 0.33,
                },
                {
                    "model": "LogisticRegression",
                    "balanced_accuracy": 0.85,
                    "f1_macro": 0.84,
                },
            ],
            "results": [],
            "training_warnings": [
                "RuntimeWarning: overflow encountered in matmul",
                "RuntimeWarning: invalid value encountered in matmul",
            ],
        },
    )

    def fake_tune_best_model(training_payload, seed=42, is_classification=True):
        tune_called["value"] = True
        return training_payload

    monkeypatch.setattr(ads, "tune_best_model", fake_tune_best_model)
    monkeypatch.setattr(
        ads,
        "evaluate_best",
        lambda training_payload, output_dir, is_classification=True: {
            "best_metrics": {
                "model": "LogisticRegression",
                "balanced_accuracy": 0.85,
                "f1_macro": 0.84,
            },
            "all_metrics": training_payload["all_metrics"],
        },
    )
    monkeypatch.setattr(
        ads,
        "reflect",
        lambda **kwargs: {
            "status": "needs_attention",
            "issues": ["Numerical instability detected."],
            "suggestions": [],
            "replan_recommended": False,
            "review_required": True,
            "training_warnings": ["RuntimeWarning: overflow encountered in matmul"],
        },
    )
    monkeypatch.setattr(
        ads,
        "derive_run_verdict",
        lambda dataset_profile, eval_payload, reflection: {
            "label": "Use with caution",
            "detail": "Training warnings require review.",
        },
    )
    monkeypatch.setattr(ads, "should_replan", lambda reflection: False)
    monkeypatch.setattr(ads, "write_markdown_report", lambda **kwargs: None)

    out_dir = agent.run(
        data_path=str(data_path),
        target="target",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )

    assert os.path.isdir(out_dir)
    assert tune_called["value"] is False

    with open(os.path.join(out_dir, "eda_summary.json")) as f:
        profile = json.load(f)
    assert any("numerical-instability" in note.lower() for note in profile.get("notes", []))
