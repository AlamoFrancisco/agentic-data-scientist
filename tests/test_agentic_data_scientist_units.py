"""
Unit tests for core logic within agentic_data_scientist.py,
specifically focusing on helper methods like _preferred_tied_model.
"""

import sys
import os
from unittest.mock import MagicMock
import pytest

# Add the project root to the sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentic_data_scientist import AgenticDataScientist


class TestPreferredTiedModel:
    """Tests for the _preferred_tied_model method."""

    @pytest.fixture
    def agent_instance(self):
        """Fixture to provide a clean AgenticDataScientist instance for each test."""
        # Use a dummy memory path as memory is not under test here
        return AgenticDataScientist(memory_path="dummy_memory.json", verbose=False)

    def test_prefers_simpler_model_when_not_significant(self, agent_instance):
        """
        Tests that the agent correctly selects the simpler model when
        there's no significant performance difference.
        """
        # Mock _model_complexity_rank to return predefined ranks
        agent_instance._model_complexity_rank = MagicMock(side_effect=lambda model_name: {
            "LogisticRegression": (1, "LogisticRegression"),
            "Ridge": (2, "Ridge"),
            "RandomForest": (4, "RandomForest"),
            "HistGradientBoosting": (6, "HistGradientBoosting"),
        }.get(model_name, (50, model_name))) # Default rank for unknown models

        # Scenario 1: LogisticRegression (rank 1) vs RandomForest (rank 4) -> LR is simpler
        reflection_1 = {
            "significance_test": {
                "significant": False,
                "model_a": "RandomForest",  # Higher rank
                "model_b": "LogisticRegression", # Lower rank (simpler)
                "p_value": 0.1,
                "note": "No significant difference."
            }
        }
        assert agent_instance._preferred_tied_model(reflection_1) == "LogisticRegression"

        # Scenario 2: Ridge (rank 2) vs LogisticRegression (rank 1) -> LR is simpler
        reflection_2 = {
            "significance_test": {
                "significant": False,
                "model_a": "Ridge", # Higher rank
                "model_b": "LogisticRegression", # Lower rank (simpler)
                "p_value": 0.1,
                "note": "No significant difference."
            }
        }
        assert agent_instance._preferred_tied_model(reflection_2) == "LogisticRegression"

    def test_returns_none_when_significant_difference(self, agent_instance):
        """Tests that no model is preferred if there's a significant difference."""
        reflection = {
            "significance_test": {
                "significant": True,
                "model_a": "RandomForest",
                "model_b": "LogisticRegression",
                "p_value": 0.01,
                "note": "Significant difference."
            }
        }
        assert agent_instance._preferred_tied_model(reflection) is None

    def test_returns_none_when_no_significance_test_info(self, agent_instance):
        """Tests that no model is preferred if significance test data is missing."""
        reflection = {} # Missing 'significance_test' key
        assert agent_instance._preferred_tied_model(reflection) is None

    def test_returns_none_when_models_are_missing_in_significance_test(self, agent_instance):
        """Tests that no model is preferred if model names are missing from the test result."""
        reflection = {
            "significance_test": {
                "significant": False,
                "p_value": 0.1,
                "note": "No significant difference."
            }
        }
        assert agent_instance._preferred_tied_model(reflection) is None