"""
Tests for meta-learning logic in agents/reflector.py
"""

import pytest
from agents.reflector import reflect


def test_meta_learning_classification_improvement():
    """Test that a classification improvement adds the correct meta-learning suggestion."""
    profile = {"is_classification": True, "imbalance_ratio": 1.0}
    evaluation = {"model": "RandomForest", "balanced_accuracy": 0.85, "f1_macro": 0.85}
    memory_hint = {
        "reflection_status": "needs_attention",
        "best_metrics": {"balanced_accuracy": 0.80}
    }
    
    result = reflect(
        dataset_profile=profile,
        evaluation=evaluation,
        all_metrics=[evaluation, {"model": "DummyClassifier", "balanced_accuracy": 0.5}],
        memory_hint=memory_hint
    )
    
    suggestions = " ".join(result["suggestions"])
    
    assert "Meta-learning: The current plan improved balanced accuracy by 0.050" in suggestions
    assert "status: needs_attention" in suggestions


def test_meta_learning_classification_degradation():
    """Test that a classification degradation raises an issue and suggests reverting."""
    profile = {"is_classification": True, "imbalance_ratio": 1.0}
    evaluation = {"model": "RandomForest", "balanced_accuracy": 0.75, "f1_macro": 0.75}
    memory_hint = {
        "reflection_status": "ok",
        "best_metrics": {"balanced_accuracy": 0.80}
    }
    
    result = reflect(
        dataset_profile=profile,
        evaluation=evaluation,
        all_metrics=[evaluation, {"model": "DummyClassifier", "balanced_accuracy": 0.5}],
        memory_hint=memory_hint
    )
    
    issues = " ".join(result["issues"])
    suggestions = " ".join(result["suggestions"])
    
    assert "Meta-learning: Balanced accuracy dropped by 0.050" in issues
    assert "Consider reverting to the strategy used in the previous run" in suggestions


def test_meta_learning_regression_improvement():
    """Test that a regression improvement adds the correct meta-learning suggestion."""
    profile = {"is_classification": False}
    evaluation = {"model": "RandomForestRegressor", "r2": 0.85}
    memory_hint = {
        "reflection_status": "needs_attention",
        "best_metrics": {"r2": 0.80}
    }
    
    result = reflect(
        dataset_profile=profile,
        evaluation=evaluation,
        all_metrics=[evaluation, {"model": "DummyRegressor", "r2": 0.1}],
        memory_hint=memory_hint
    )
    
    suggestions = " ".join(result["suggestions"])
    
    assert "Meta-learning: The current plan improved R² by 0.050" in suggestions


def test_meta_learning_regression_degradation():
    """Test that a regression degradation raises an issue and suggests reverting."""
    profile = {"is_classification": False}
    evaluation = {"model": "RandomForestRegressor", "r2": 0.75}
    memory_hint = {
        "reflection_status": "ok",
        "best_metrics": {"r2": 0.80}
    }
    
    result = reflect(
        dataset_profile=profile,
        evaluation=evaluation,
        all_metrics=[evaluation, {"model": "DummyRegressor", "r2": 0.1}],
        memory_hint=memory_hint
    )
    
    issues = " ".join(result["issues"])
    suggestions = " ".join(result["suggestions"])
    
    assert "Meta-learning: R² dropped by 0.050" in issues
    assert "Consider reverting to the strategy used in the previous run" in suggestions