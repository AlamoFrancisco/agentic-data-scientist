"""
Tests for agents/memory.py

Covers: now_iso, JSONMemory._load, save, get_dataset_record,
upsert_dataset_record, add_note, size_bucket, _similarity_score,
get_similar_record.
"""
import json
import os
import pytest

from agents.memory import JSONMemory, now_iso


# ── now_iso ──────────────────────────────────────────────────────────────────

def test_now_iso_format():
    ts = now_iso()
    assert ts.endswith("Z")
    assert "T" in ts


# ── _load / init ─────────────────────────────────────────────────────────────

def test_load_creates_empty_when_file_missing(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    assert m.data == {"datasets": {}, "notes": []}


def test_load_reads_existing_json(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text(json.dumps({"datasets": {"fp1": {"shape": {"rows": 100}}}, "notes": []}))
    m = JSONMemory(path=str(path))
    assert "fp1" in m.data["datasets"]


def test_load_corrupt_json_creates_backup(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("not json {{{")
    m = JSONMemory(path=str(path))
    assert os.path.exists(str(path) + ".bak")
    assert m.data["datasets"] == {}


# ── save ─────────────────────────────────────────────────────────────────────

def test_save_writes_json(tmp_path):
    path = tmp_path / "mem.json"
    m = JSONMemory(path=str(path))
    m.data["datasets"]["fp1"] = {"foo": "bar"}
    m.save()
    loaded = json.loads(path.read_text())
    assert "fp1" in loaded["datasets"]


# ── get / upsert ─────────────────────────────────────────────────────────────

def test_get_dataset_record_returns_existing(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    m.data["datasets"]["fp1"] = {"best_model": "RF"}
    assert m.get_dataset_record("fp1") == {"best_model": "RF"}


def test_get_dataset_record_missing_returns_none(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    assert m.get_dataset_record("nonexistent") is None


def test_get_dataset_record_require_reliable_skips_non_reliable_record(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    m.data["datasets"]["fp1"] = {"best_model": "RF", "verdict_label": "Use with caution"}
    assert m.get_dataset_record("fp1", require_reliable=True) is None


def test_get_dataset_record_require_reliable_returns_reliable_record(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    m.data["datasets"]["fp1"] = {"best_model": "RF", "verdict_label": "Reliable result"}
    assert m.get_dataset_record("fp1", require_reliable=True) == {"best_model": "RF", "verdict_label": "Reliable result"}


def test_upsert_persists_and_is_retrievable(tmp_path):
    path = tmp_path / "mem.json"
    m = JSONMemory(path=str(path))
    m.upsert_dataset_record("fp1", {"best_model": "LR"})
    assert m.get_dataset_record("fp1") == {"best_model": "LR"}
    # Verify it's written to disk
    m2 = JSONMemory(path=str(path))
    assert m2.get_dataset_record("fp1") == {"best_model": "LR"}


# ── add_note ─────────────────────────────────────────────────────────────────

def test_add_note_stores_message(tmp_path):
    path = tmp_path / "mem.json"
    m = JSONMemory(path=str(path))
    m.add_note("test note")
    assert any("test note" in n["msg"] for n in m.data["notes"])


# ── size_bucket ───────────────────────────────────────────────────────────────

def test_size_bucket_small():
    m = JSONMemory.__new__(JSONMemory)
    assert m.size_bucket(500) == "small"
    assert m.size_bucket(999) == "small"


def test_size_bucket_medium():
    m = JSONMemory.__new__(JSONMemory)
    assert m.size_bucket(1000) == "medium"
    assert m.size_bucket(5000) == "medium"


def test_size_bucket_large():
    m = JSONMemory.__new__(JSONMemory)
    assert m.size_bucket(10000) == "large"
    assert m.size_bucket(100000) == "large"


# ── _similarity_score ────────────────────────────────────────────────────────

def test_similarity_score_same_bucket_gives_positive(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}}
    record  = {"shape": {"rows": 700}, "imbalance_ratio": 1.0, "missing_pct": {}}
    assert m._similarity_score(profile, record) > 0


def test_similarity_score_different_bucket_penalised(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500},   "imbalance_ratio": 1.0, "missing_pct": {}}
    record  = {"shape": {"rows": 50000}, "imbalance_ratio": 1.0, "missing_pct": {}}
    # Different size buckets → lower score than same bucket
    s_same = m._similarity_score(profile, {"shape": {"rows": 700}, "imbalance_ratio": 1.0, "missing_pct": {}})
    s_diff = m._similarity_score(profile, record)
    assert s_diff < s_same


# ── get_similar_record ────────────────────────────────────────────────────────

def test_get_similar_record_returns_match_above_threshold(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}}
    m.data["datasets"]["fp1"] = {
        "shape": {"rows": 700}, "imbalance_ratio": 1.0, "missing_pct": {},
        "best_model": "RF",
    }
    result = m.get_similar_record(profile, threshold=0.5)
    assert result is not None
    assert result["best_model"] == "RF"


def test_get_similar_record_returns_none_below_threshold(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500},   "imbalance_ratio": 1.0, "missing_pct": {}}
    # Very different record — different size bucket AND different imbalance
    m.data["datasets"]["fp1"] = {
        "shape": {"rows": 50000}, "imbalance_ratio": 5.0, "missing_pct": {},
    }
    result = m.get_similar_record(profile, threshold=0.9)
    assert result is None


def test_get_similar_record_empty_memory(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}}
    assert m.get_similar_record(profile) is None
