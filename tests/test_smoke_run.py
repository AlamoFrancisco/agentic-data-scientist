import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
