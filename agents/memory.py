import json
import os
import shutil
from typing import Any, Dict, Optional
from datetime import datetime


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

    def get_dataset_record(self, fingerprint: str, dataset_name: Optional[str] = None, target: Optional[str] = None, shape: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        datasets = self.data.get("datasets", {})
        # 1. Exact fingerprint match (same filename)
        if fingerprint in datasets:
            return datasets[fingerprint]
        # 2. Same dataset name
        if dataset_name:
            for record in datasets.values():
                if record.get("dataset") == dataset_name:
                    return record
        # 3. Same target + shape (renamed file)
        if target and shape:
            for record in datasets.values():
                if record.get("target") == target and record.get("shape") == shape:
                    return record
        return None

    def upsert_dataset_record(self, fingerprint: str, record: Dict[str, Any]) -> None:
        datasets = self.data.setdefault("datasets", {})
        # If a record with the same target + shape already exists, update it in place
        target = record.get("target")
        shape = record.get("shape")
        for existing_fp, existing_record in datasets.items():
            if existing_record.get("target") == target and existing_record.get("shape") == shape:
                datasets[existing_fp] = record
                self.save()
                return
        # No match — new dataset, store under the fingerprint key
        datasets[fingerprint] = record
        self.save()

    def add_note(self, msg: str) -> None:
        self.data.setdefault("notes", []).append({"ts": now_iso(), "msg": msg})
        self.save()
        
    def get_similar_record(self, profile: Dict[str, Any], threshold: float = 0.5) -> Optional[Dict[str, Any]]:
        # Start with no match found and a score of zero
        best_match = None
        best_score = 0.0
        
        # Loop through all stored dataset records in memory
        for fingerprint, record in self.data.get("datasets", {}).items():
            # Calculate how similar this stored record is to the current profile
            score = self._similarity_score(profile, record)
            # Only update if this is the best match so far AND above the threshold
            if score > best_score and score >= threshold:
                best_score = score
                best_match = record
        
        # Return the best match found, or None if nothing was similar enough
        return best_match
    
    def size_bucket(self, rows: int) -> str:
        if rows < 1000:
            return "small"
        elif rows < 10000:
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
        if (current_imb >= 3.0) == (record_imb >= 3.0):
            score += 1
        checks += 1

        # Check 3: similar missing data? — both have high or low missingness
        current_missing = max(profile.get("missing_pct", {}).values(), default=0)
        record_missing = record.get("missing_pct", 0)
        if isinstance(record_missing, dict):
            record_missing = max(record_missing.values(), default=0)
        if (current_missing > 20) == (record_missing > 20):
            score += 1
        checks += 1
        
        # Return average score across all checks, or 0 if no checks were done
        return score / checks if checks > 0 else 0.0

   

