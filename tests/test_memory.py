from agents.memory import JSONMemory

memory = JSONMemory()

profile = {"shape": {"rows": 500}, "imbalance_ratio": 2.0, "missing_pct": {}}

memory.data["datasets"]["fake_fp"] = {
    "shape": {"rows": 800},
    "imbalance_ratio": 2.5,
    "missing_pct": {}
}

result = memory.get_similar_record(profile)
print(result)

