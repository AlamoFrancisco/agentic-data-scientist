from agents.reflector import reflect

# Fake a scenario where overfitting should be detected
profile = {"imbalance_ratio": 1.0, "missing_pct": {}}
evaluation = {
    "model": "RandomForest",
    "balanced_accuracy": 0.95,
    "f1_macro": 0.55
}
all_metrics = [
    {"model": "RandomForest", "f1_macro": 0.55},
    {"model": "LogisticRegression", "f1_macro": 0.54},
    {"model": "DummyClassifier", "balanced_accuracy": 0.91, "f1_macro": 0.50},
]

result = reflect(profile, evaluation, all_metrics)
print("Status:", result["status"])
print("Issues:", result["issues"])
print("Suggestions:", result["suggestions"])
print("Replan:", result["replan_recommended"])
