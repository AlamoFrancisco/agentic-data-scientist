"""
Memory Agent

Persistent JSON-backed store that lets the agent learn from prior runs.

Implemented:
- Dataset record storage keyed by fingerprint (SHA-256 of shape + column names + target)
- Multi-strategy lookup: exact fingerprint → dataset filename → target + shape
- Reliable-record gate: only successful runs (verdict_label == "Reliable result")
  are reused as planning hints; failed runs are stored as diagnostics only
- Failed-target tracking: targets that produced no useful result are skipped on
  subsequent auto-detect runs for the same dataset
- Cross-dataset similarity matching via size bucket, imbalance flag, and
  missingness level — used to surface hints for unseen datasets

TODO:
- Meta-learning from reflection history (track which suggestions led to improvement)
- Time-decay on stored records (stale results should carry less weight)
"""

import json
import os
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime

from config import (
    IMBALANCE_THRESHOLD,
    LARGE_DATASET_ROWS,
    SEVERE_MISSING_THRESHOLD,
    SMALL_DATASET_ROWS,
)


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class JSONMemory:
    """
    Lightweight persistent memory for the agent.
    Stores dataset fingerprint -> best model/metrics and notes.
    """

    def __init__(self, path: str = "agent_memory.json"):
        self.path = path
        self.data: Dict[str, Any] = {"datasets": {}, "notes": []}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            backup = self.path + ".bak"
            shutil.copy(self.path, backup)
            self.data = {"datasets": {}, "notes": [{"ts": now_iso(), "msg": f"Memory reset; backup at {backup}"}]}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def _is_reliable_record(self, record: Optional[Dict[str, Any]]) -> bool:
        return bool(record) and record.get("verdict_label") == "Reliable result"

    def _matches_target_origins(
        self,
        record: Optional[Dict[str, Any]],
        allowed_target_origins: Optional[List[str]],
    ) -> bool:
        if not allowed_target_origins:
            return True
        if not record:
            return False
        origin = record.get("target_origin") or record.get("target_source")
        return origin in set(allowed_target_origins)

    def get_dataset_record(
        self,
        fingerprint: str,
        dataset_name: Optional[str] = None,
        target: Optional[str] = None,
        shape: Optional[Dict] = None,
        require_reliable: bool = False,
        allowed_target_origins: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        datasets = self.data.get("datasets", {})
        # 1. Exact fingerprint match (same filename)
        if fingerprint in datasets:
            record = datasets[fingerprint]
            if (not require_reliable or self._is_reliable_record(record)) and self._matches_target_origins(record, allowed_target_origins):
                return record
        # 2. Same dataset name
        if dataset_name:
            for record in datasets.values():
                if (
                    record.get("dataset") == dataset_name
                    and (not require_reliable or self._is_reliable_record(record))
                    and self._matches_target_origins(record, allowed_target_origins)
                ):
                    return record
        # 3. Same target + shape (renamed file)
        if target and shape:
            for record in datasets.values():
                if (
                    record.get("target") == target
                    and record.get("shape") == shape
                    and (not require_reliable or self._is_reliable_record(record))
                    and self._matches_target_origins(record, allowed_target_origins)
                ):
                    return record
        return None

    def upsert_dataset_record(self, fingerprint: str, record: Dict[str, Any]) -> None:
        datasets = self.data.setdefault("datasets", {})
        # If a record with the same target + shape already exists, update it in place
        target = record.get("target")
        shape = record.get("shape")
        
        old_fp = None
        if target and shape:
            for existing_fp, existing_record in datasets.items():
                if existing_record.get("target") == target and existing_record.get("shape") == shape:
                    old_fp = existing_fp
                    break
        if old_fp and old_fp != fingerprint:
            del datasets[old_fp]
            
        # No match — new dataset, store under the fingerprint key
        datasets[fingerprint] = record
        self.save()

    def add_note(self, msg: str) -> None:
        self.data.setdefault("notes", []).append({"ts": now_iso(), "msg": msg})
        self.save()

    def add_failed_target(self, dataset_name: str, target: str) -> None:
        """Record a target that produced no useful results on a dataset."""
        failed = self.data.setdefault("failed_targets", {})
        targets = failed.setdefault(dataset_name, [])
        if target not in targets:
            targets.append(target)
        self.save()

    def get_failed_targets(self, dataset_name: str) -> List[str]:
        """Return list of targets that previously failed on this dataset."""
        return self.data.get("failed_targets", {}).get(dataset_name, [])
        
    def get_similar_record(self, profile: Dict[str, Any], threshold: float = 0.5) -> Optional[Dict[str, Any]]:
        # Start with no match found and a score of zero
        best_match = None
        best_score = 0.0
        
        # Loop through all stored dataset records in memory
        for fingerprint, record in self.data.get("datasets", {}).items():
            # Only transfer knowledge from successful prior runs
            if not self._is_reliable_record(record):
                continue
                
            # Calculate how similar this stored record is to the current profile
            score = self._similarity_score(profile, record)
            # Only update if this is the best match so far AND above the threshold
            if score > best_score and score >= threshold:
                best_score = score
                best_match = record
        
        # Return the best match found, or None if nothing was similar enough
        return best_match
    
    def size_bucket(self, rows: int) -> str:
        if rows < SMALL_DATASET_ROWS:
            return "small"
        elif rows < LARGE_DATASET_ROWS:
            return "medium"
        else:
            return "large"

    def _similarity_score(self, profile: Dict[str, Any], record: Dict[str, Any]) -> float:
        # Accumulators to build the final average score
        score = 0.0
        checks = 0
        
        # Check 1: similar size? — same size bucket means similar data volume
        current_rows = profile.get("shape", {}).get("rows", 0)
        record_rows = record.get("shape", {}).get("rows", 0)

        if self.size_bucket(current_rows) == self.size_bucket(record_rows):
            score += 1
        checks += 1

        # Check 2: both imbalanced? — same imbalance level means similar class distribution
        current_imb = profile.get("imbalance_ratio") or 1.0
        record_imb = record.get("imbalance_ratio") or 1.0
        if (current_imb >= IMBALANCE_THRESHOLD) == (record_imb >= IMBALANCE_THRESHOLD):
            score += 1
        checks += 1

        # Check 3: similar missing data? — both have high or low missingness
        current_missing = max(profile.get("missing_pct", {}).values(), default=0)
        record_missing = record.get("missing_pct", 0)
        if isinstance(record_missing, dict):
            record_missing = max(record_missing.values(), default=0)
        if (current_missing > SEVERE_MISSING_THRESHOLD) == (record_missing > SEVERE_MISSING_THRESHOLD):
            score += 1
        checks += 1
        
        # Return average score across all checks, or 0 if no checks were done
        return score / checks if checks > 0 else 0.0

   
