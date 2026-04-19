"""
Tests for agents/memory.py

Covers: now_iso, JSONMemory._load, save, get_dataset_record,
upsert_dataset_record, add_note, size_bucket, _similarity_score,
get_similar_record.
"""
import json
import os
import pytest
from datetime import datetime, timedelta

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


def test_get_dataset_record_allowed_target_origins_filters_records(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    m.data["datasets"]["fp1"] = {
        "best_model": "RF",
        "verdict_label": "Reliable result",
        "target_source": "inferred",
        "target_origin": "inferred",
    }
    assert m.get_dataset_record("fp1", require_reliable=True, allowed_target_origins=["manual"]) is None


def test_get_dataset_record_allowed_target_origins_accepts_manual_record(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    m.data["datasets"]["fp1"] = {
        "best_model": "RF",
        "verdict_label": "Reliable result",
        "target_source": "memory",
        "target_origin": "manual",
    }
    assert m.get_dataset_record("fp1", require_reliable=True, allowed_target_origins=["manual"]) == {
        "best_model": "RF",
        "verdict_label": "Reliable result",
        "target_source": "memory",
        "target_origin": "manual",
    }


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


def test_similarity_score_time_decay_recent(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}}
    
    # Recent record (0 days old)
    recent_ts = datetime.utcnow().isoformat() + "Z"
    record = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}, "last_seen": recent_ts}
    
    score = m._similarity_score(profile, record)
    assert score == 1.0  # All checks pass, no decay


def test_similarity_score_time_decay_half_life(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}}
    
    # 90 days old
    old_ts = (datetime.utcnow() - timedelta(days=90)).isoformat() + "Z"
    record = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}, "last_seen": old_ts}
    
    score = m._similarity_score(profile, record)
    assert 0.49 <= score <= 0.51  # Approx 0.5


def test_similarity_score_time_decay_capped(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}}
    
    # 200 days old (cap is at 0.5)
    ancient_ts = (datetime.utcnow() - timedelta(days=200)).isoformat() + "Z"
    record = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}, "last_seen": ancient_ts}
    
    score = m._similarity_score(profile, record)
    assert score == 0.5  # Hard cap at 0.5


def test_similarity_score_time_decay_invalid_date(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}}
    
    record = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}, "last_seen": "not-a-date"}
    
    score = m._similarity_score(profile, record)
    assert score == 1.0  # Fails gracefully, ignores decay


# ── get_similar_record ────────────────────────────────────────────────────────

def test_get_similar_record_returns_match_above_threshold(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}}
    m.data["datasets"]["fp1"] = {
        "shape": {"rows": 700}, "imbalance_ratio": 1.0, "missing_pct": {},
        "best_model": "RF", "verdict_label": "Reliable result",
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
        "verdict_label": "Reliable result",
    }
    result = m.get_similar_record(profile, threshold=0.9)
    assert result is None


def test_get_similar_record_empty_memory(tmp_path):
    m = JSONMemory(path=str(tmp_path / "mem.json"))
    profile = {"shape": {"rows": 500}, "imbalance_ratio": 1.0, "missing_pct": {}}
    assert m.get_similar_record(profile) is None
