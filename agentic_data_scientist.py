"""
AgenticDataScientist — Orchestrator

End-to-end offline data science pipeline: load → profile → plan → train →
evaluate → reflect → (replan) → persist.

Implemented:
- Auto target detection: checks memory for a prior reliable result before
  inferring; skips targets recorded as failed on the same dataset
- Planning loop: Planner produces an adaptive plan; Executor interprets each
  flag and sets profile keys consumed by build_preprocessor / select_models
- Training retry: up to 3 attempts before aborting the run
- Cross-validation: triggered when validate_with_cross_validation is in the plan;
  top-k candidates evaluated with StratifiedKFold / KFold
- Reflection and replanning: Reflector analyses results; if replan_recommended
  and max_replans not reached, apply_replan_strategy modifies plan + profile
  and the loop restarts
- Memory write-back: reliable runs stored under dataset fingerprint for future
  prioritisation; failed targets and invalid runs stored as diagnostics
- Per-run artefacts: eda_summary.json, plan.json, metrics.json, reflection.json,
  report.md, confusion_matrix.png / predicted_vs_actual.png
"""
import os
import json
import inspect
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Agent components and tooling used by the orchestrator
from config import (
    SMALL_DATASET_ROWS,
    BASELINE_MIN_IMPROVEMENT_CLS,
    BASELINE_MIN_IMPROVEMENT_REG,
    NEAR_PERFECT_THRESHOLD,
    R2_LOW_THRESHOLD,
    CV_TOP_K,
    CV_TOP_K_SMALL,
)
from agents.planner import create_plan, apply_replan_strategy
from agents.reflector import reflect, should_replan
from agents.memory import JSONMemory
from tools.data_profiler import profile_dataset, infer_target_column, dataset_fingerprint, is_classification_target
from tools.modelling import build_preprocessor, cross_validate_top_models, select_models, train_models, tune_best_model
from tools.evaluation import evaluate_best, write_markdown_report, save_json, derive_run_verdict


# Lightweight container for run metadata and parameters
@dataclass
class RunContext:
    run_id: str
    started_at: str
    data_path: str
    target: str
    output_dir: str
    seed: int
    test_size: float
    max_replans: int


def now_iso() -> str:
    """Return current UTC time in ISO 8601 format (no microseconds) with Z suffix."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class AgenticDataScientist:
    """
    Offline Agentic Data Scientist (classification-focused).

    Responsibilities:
    - load and profile datasets
    - create a plan (via planner)
    - build preprocessors and select candidate models
    - train and evaluate models
    - reflect on results and optionally re-plan
    - persist artefacts and update memory
    """

    def __init__(self, memory_path: str = "agent_memory.json", verbose: bool = True):
        # Verbose controls logging output
        self.verbose = verbose
        # Simple persistent memory used to remember prior runs for a dataset fingerprint
        self.memory = JSONMemory(memory_path)

        # Context and transient state populated when run() is executed
        self.ctx: Optional[RunContext] = None
        self.state: Dict[str, Any] = {}

    def log(self, msg: str) -> None:
        """Print a log message when verbose is enabled."""
        if self.verbose:
            print(f"[AgenticDataScientist] {msg}")

    def _format_decision_summary(self, plan: List[str], profile: Dict[str, Any]) -> List[str]:
        decisions: List[str] = []

        if "apply_oversampling" in plan:
            profile["apply_oversampling"] = True
            decisions.append("oversample minority class (SMOTE)")

        if "consider_imbalance_strategy" in plan:
            profile["use_class_weights"] = True
            decisions.append("imbalance-aware class weights")

        if "apply_regularization" in plan:
            profile["use_regularization"] = True
            decisions.append("stronger regularization")

        if "handle_severe_missing_data" in plan:
            profile["robust_imputation"] = True
            decisions.append("robust imputation")

        if "apply_target_encoding" in plan:
            profile["use_target_encoding"] = True
            decisions.append("target encoding")

        if "apply_feature_engineering" in plan:
            profile["use_feature_engineering"] = True
            decisions.append("feature engineering")

        if "apply_robust_scaling" in plan:
            profile["use_robust_scaling"] = True
            decisions.append("robust scaling")

        if "handle_outliers" in plan:
            profile["handle_outliers"] = True
            decisions.append("outlier-aware preprocessing")

        if "drop_near_constant_features" in plan:
            near_const = profile.get("near_constant_cols", [])
            decisions.append(f"drop {len(near_const)} near-constant feature(s)")

        if "drop_correlated_features" in plan:
            corr_cols = profile.get("corr_cols_to_drop", [])
            profile["drop_high_corr"] = True
            decisions.append(f"drop {len(corr_cols)} correlated feature(s)")

        if "drop_leaky_features" in plan:
            leaky_cols = profile.get("leaky_col_names", [])
            profile["drop_leaky"] = True
            if leaky_cols:
                decisions.append(f"drop leaky features: {leaky_cols}")
            else:
                decisions.append("drop suspected leaky features")
                
        if "drop_sensitive_features" in plan:
            sensitive_cols = profile.get("sensitive_cols", [])
            profile["drop_sensitive"] = True
            decisions.append(f"drop sensitive features: {sensitive_cols}")

        if "use_simple_models_only" in plan:
            profile["simple_models_only"] = True
            decisions.append("simple models only")

        if "use_ensemble_models" in plan:
            profile["prefer_ensemble"] = True
            decisions.append("favor ensemble models")

        if "tune_hyperparameters" in plan:
            decisions.append("hyperparameter tuning")
            
        if "reduce_tuning_budget" in plan:
            profile["reduce_tuning_budget"] = True
            decisions.append("reduce tuning budget (cost-aware)")

        preferred_model = self._preferred_model_from_plan(plan)
        if preferred_model:
            decisions.append(f"memory priority: {preferred_model}")

        return decisions

    def _plan_headline(self, plan: List[str]) -> str:
        labels: List[str] = []

        if "profile_dataset" in plan:
            labels.append("profiling")

        preprocessing_flags = {
            "handle_severe_missing_data",
            "apply_robust_scaling",
            "handle_outliers",
            "apply_target_encoding",
            "drop_near_constant_features",
            "drop_correlated_features",
            "apply_feature_engineering",
        }
        if any(step in plan for step in preprocessing_flags):
            labels.append("robust preprocessing")

        if "drop_leaky_features" in plan and "drop_sensitive_features" in plan:
            labels.append("leak & bias handling")
        elif "drop_leaky_features" in plan:
            labels.append("leak handling")
        elif "drop_sensitive_features" in plan:
            labels.append("bias handling")

        if "use_simple_models_only" in plan:
            labels.append("simple models")
        elif "use_ensemble_models" in plan:
            labels.append("ensemble models")
        elif "select_models" in plan:
            labels.append("candidate models")

        if "tune_hyperparameters" in plan:
            labels.append("hyperparameter tuning")

        if "validate_with_cross_validation" in plan:
            labels.append("cross-validation")

        if "reflect" in plan:
            labels.append("reflection")

        if not labels:
            return f"{len(plan)} steps"
        return ", ".join(labels)

    def _log_execution_pass(self, pass_index: int, total_passes: int, decisions: List[str]) -> None:
        self.log(f"Execution pass {pass_index}/{total_passes}")
        if not decisions:
            return

        summary = "; ".join(decisions)
        if self.state.get("last_decision_summary") == summary:
            self.log("Decision summary unchanged from previous pass.")
        else:
            self.log(f"Decision summary: {summary}")
            self.state["last_decision_summary"] = summary

    def _log_replan_diff(self, old_plan: List[str], new_plan: List[str], reflection: Dict[str, Any]) -> None:
        ignored_steps = {"replan_attempt"}
        added = [step for step in new_plan if step not in old_plan and step not in ignored_steps]
        removed = [step for step in old_plan if step not in new_plan and step not in ignored_steps]
        reason = (reflection.get("issues") or ["reflection requested follow-up"])[0]

        changes: List[str] = []
        if added:
            changes.append(f"added {added}")
        if removed:
            changes.append(f"removed {removed}")
        if not changes:
            changes.append("no material strategy changes")

        self.log(f"Replan changes: {'; '.join(changes)}")
        self.log(f"Replan reason: {reason}")

    def _log_final_summary(self, eval_payload: Dict[str, Any], verdict: Dict[str, str]) -> None:
        best = eval_payload.get("best_metrics", {})
        model_name = best.get("model", "Unknown")
        is_classification = best.get("balanced_accuracy") is not None

        if is_classification:
            metric_text = (
                f"balanced accuracy={float(best.get('balanced_accuracy', 0.0)):.3f}, "
                f"macro F1={float(best.get('f1_macro', 0.0)):.3f}"
            )
        else:
            metric_text = (
                f"R²={float(best.get('r2', 0.0)):.3f}, "
                f"MAE={float(best.get('mae', 0.0)):.3f}, "
                f"RMSE={float(best.get('rmse', 0.0)):.3f}"
            )

        cv_payload = eval_payload.get("cross_validation") or {}
        cv_text = "not run"
        if cv_payload.get("enabled"):
            cv_entry = next(
                (item for item in cv_payload.get("models", []) if item.get("model") == model_name),
                None,
            )
            if cv_entry:
                if is_classification:
                    cv_text = (
                        f"{cv_payload.get('n_splits', 0)} folds, "
                        f"balanced accuracy={float(cv_entry.get('balanced_accuracy_mean', 0.0)):.3f} +/- "
                        f"{float(cv_entry.get('balanced_accuracy_std', 0.0)):.3f}"
                    )
                else:
                    cv_text = (
                        f"{cv_payload.get('n_splits', 0)} folds, "
                        f"R²={float(cv_entry.get('r2_mean', 0.0)):.3f} +/- "
                        f"{float(cv_entry.get('r2_std', 0.0)):.3f}"
                    )
            else:
                cv_text = f"{cv_payload.get('n_splits', 0)} folds completed"

        self.log(
            "Final summary:\n"
            f"  Best model: {model_name} ({metric_text})\n"
            f"  Cross-validation: {cv_text}\n"
            f"  Verdict: {verdict['label']} — {verdict['detail']}\n"
            f"  Outputs: {self.ctx.output_dir}"
        )

    def _ordered_auto_target_candidates(
        self,
        scores: Dict[str, float],
        failed_targets: List[str],
        max_candidates: int = 3,
    ) -> List[str]:
        failed = set(failed_targets)
        ordered = [
            column
            for column, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if score > 0 and column not in failed
        ]
        return ordered[:max_candidates]

    def _auto_target_is_unsuitable(
        self,
        eval_payload: Dict[str, Any],
        verdict: Dict[str, str],
        is_classification: bool,
    ) -> bool:
        if verdict.get("label") == "Invalid due to leakage risk":
            return True

        best = eval_payload.get("best_metrics", {})
        best_model = str(best.get("model", ""))
        if "Dummy" in best_model:
            return True

        all_metrics = eval_payload.get("all_metrics", [])
        baseline = next((m for m in all_metrics if "Dummy" in str(m.get("model", ""))), None)

        if is_classification:
            best_score = float(best.get("balanced_accuracy", 0.0))
            baseline_score = float((baseline or {}).get("balanced_accuracy", 0.0))
            return (best_score - baseline_score) < BASELINE_MIN_IMPROVEMENT_CLS

        best_score = float(best.get("r2", 0.0))
        baseline_score = float((baseline or {}).get("r2", 0.0))
        if best_score < R2_LOW_THRESHOLD:
            return True
        return (best_score - baseline_score) < BASELINE_MIN_IMPROVEMENT_REG

    def _tuning_skip_reason(
        self,
        training_payload: Dict[str, Any],
        is_classification: bool,
    ) -> Optional[str]:
        best = training_payload.get("best", {})
        best_metrics = best.get("metrics", {})
        best_model = str(best.get("name") or best_metrics.get("model", ""))
        if "Dummy" in best_model:
            return "best candidate is still the dummy baseline"

        warning_keywords = ("overflow", "divide by zero", "invalid value", "singular", "nan")
        training_warnings = [str(w).lower() for w in training_payload.get("training_warnings", [])]
        if any(keyword in warning for warning in training_warnings for keyword in warning_keywords):
            return "initial training already showed numerical-instability warnings"

        baseline = next(
            (m for m in training_payload.get("all_metrics", []) if "Dummy" in str(m.get("model", ""))),
            None,
        )
        if not baseline:
            return None

        if is_classification:
            best_score = float(best_metrics.get("balanced_accuracy", 0.0))
            if best_score >= NEAR_PERFECT_THRESHOLD:
                return f"untuned balanced accuracy is already near-perfect ({best_score:.3f})"
            baseline_score = float(baseline.get("balanced_accuracy", 0.0))
            improvement = best_score - baseline_score
            if improvement < BASELINE_MIN_IMPROVEMENT_CLS:
                return (
                    f"best model only {improvement:.3f} better than dummy baseline "
                    f"({best_score:.3f} vs {baseline_score:.3f})"
                )
            return None

        best_score = float(best_metrics.get("r2", 0.0))
        if best_score >= NEAR_PERFECT_THRESHOLD:
            return f"untuned R² is already near-perfect ({best_score:.3f})"
        baseline_score = float(baseline.get("r2", 0.0))
        improvement = best_score - baseline_score
        if best_score < R2_LOW_THRESHOLD:
            return f"best model R² ({best_score:.3f}) is still below the useful-signal threshold"
        if improvement < BASELINE_MIN_IMPROVEMENT_REG:
            return (
                f"best model only {improvement:.3f} better than dummy baseline "
                f"(R² {best_score:.3f} vs {baseline_score:.3f})"
            )
        return None

    def load_data(self, path: str) -> pd.DataFrame:
        """Load a CSV into a pandas DataFrame and log its shape."""
        self.log(f"Loading dataset: {path}")
        df = pd.read_csv(path, na_values=["?", "NA", "N/A", "NULL", ""])
        self.log(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        self.state["original_row_count"] = before
        self.state["duplicate_count"] = dropped
        if dropped:
            self.log(f"Dropped {dropped} duplicate rows ({dropped/before*100:.1f}%)")
        return df

    def _preferred_model_from_plan(self, plan: List[str]) -> Optional[str]:
        """Extract a memory-driven model priority from the plan, if present."""
        for step in plan:
            if step.startswith("prioritize_model:"):
                _, _, model_name = step.partition(":")
                return model_name or None
        return None

    def _evaluate_best_compat(
        self,
        training_payload: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call evaluate_best while remaining compatible with older test doubles.

        The production evaluate_best accepts dataset_profile, but several smoke
        tests replace it with lambdas that still use the earlier signature.
        """
        kwargs: Dict[str, Any] = {
            "output_dir": self.ctx.output_dir,
            "is_classification": profile.get("is_classification", True),
        }

        try:
            signature = inspect.signature(evaluate_best)
        except (TypeError, ValueError):
            signature = None

        if signature is None:
            kwargs["dataset_profile"] = profile
        else:
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if accepts_kwargs or "dataset_profile" in signature.parameters:
                kwargs["dataset_profile"] = profile

        return evaluate_best(training_payload, **kwargs)

    def run(
        self,
        data_path: str,
        target: str,
        output_root: str = "outputs",
        seed: int = 42,
        test_size: float = 0.2,
        max_replans: int = 1,
    ) -> str:
        """
        Main orchestration entry point.

        Parameters:
        - data_path: path to the CSV dataset
        - target: target column name or 'auto' to infer
        - output_root: directory where outputs are stored (subdir will be created)
        - seed/test_size: training reproducibility and test split
        - max_replans: maximum number of times to re-plan and re-run

        Returns: path to the output directory for this run
        """
        # Create a unique run id and output directory for artefacts
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

        # Populate run context with parameters and metadata
        self.ctx = RunContext(
            run_id=run_id,
            started_at=now_iso(),
            data_path=os.path.basename(data_path),
            target=target,
            output_dir=output_dir,
            seed=seed,
            test_size=test_size,
            max_replans=max_replans,
        )
        # Internal state used to track replanning attempts
        self.state = {"replan_count": 0}

        # Load dataset into memory
        df = self.load_data(data_path)

        # If client requested auto target detection, check memory first before inferring
        target_source = "manual"
        target_origin = "manual"
        target_candidate_scores = None
        auto_target_candidates: List[str] = []
        if target.strip().lower() == "auto":
            prev_hint = self.memory.get_dataset_record(
                "",
                dataset_name=self.ctx.data_path,
                require_reliable=True,
                allowed_target_origins=["manual"],
            )
            failed = self.memory.get_failed_targets(self.ctx.data_path)
            if prev_hint and prev_hint.get("target") and prev_hint["target"] not in failed:
                stored_target = prev_hint["target"]
                self.ctx.target = stored_target
                target_source = "memory"
                target_origin = prev_hint.get("target_origin") or prev_hint.get("target_source") or "memory"
                self.log(f"Using target from memory: '{stored_target}'")
            else:
                inferred, target_candidate_scores = infer_target_column(df, return_scores=True)
                auto_target_candidates = self._ordered_auto_target_candidates(
                    target_candidate_scores or {},
                    failed,
                )
                if not auto_target_candidates:
                    raise ValueError("Could not infer target column. Please provide --target <name>.")
                self.ctx.target = auto_target_candidates[0]
                target_source = "inferred"
                target_origin = "inferred"
                target_type = "classification" if is_classification_target(df[self.ctx.target]) else "regression"
                self.log(f"Inferred target: {self.ctx.target} (type: {target_type})")

        eval_payload: Dict[str, Any] = {}
        verdict: Dict[str, str] = {}
        profile: Dict[str, Any] = {}
        while True:
            # Produce a dataset profile (EDA summary) and a fingerprint used for memory
            profile = profile_dataset(
                df,
                self.ctx.target,
                target_source=target_source,
                target_candidate_scores=target_candidate_scores,
                duplicate_count=self.state.get("duplicate_count"),
                original_row_count=self.state.get("original_row_count"),
            )
            profile["dataset"] = self.ctx.data_path
            fp = dataset_fingerprint(df, self.ctx.target, file_path=self.ctx.data_path)

            # Look up previous runs — by fingerprint, then filename, then target+shape
            prev = self.memory.get_dataset_record(
                fp,
                dataset_name=self.ctx.data_path,
                target=self.ctx.target,
                shape=profile["shape"],
                require_reliable=True,
            )
            if prev:
                self.log(f"Memory hit: previously best={prev.get('best_model')} for fp={fp}")
            else:
                # No exact match — try cross-dataset similarity as a weaker hint.
                # Guard: only use a similar record from the same task type to avoid
                # passing a classification model name into a regression candidate list.
                similar = self.memory.get_similar_record(profile)
                if similar and similar.get("is_classification") == profile.get("is_classification"):
                    self.log(
                        f"Cross-dataset similarity hint: structurally similar dataset used "
                        f"{similar.get('best_model')} — using as soft planning hint."
                    )
                    prev = similar

            # Create an initial plan informed by the profile and optional memory hint
            plan = create_plan(profile, memory_hint=prev)
            self.log(f"Plan ready: {self._plan_headline(plan)}.")

            # Execution loop: trains and evaluates, then optionally replans and repeats
            while True:
                pass_index = self.state["replan_count"] + 1
                total_passes = self.ctx.max_replans + 1
                decisions = self._format_decision_summary(plan, profile)
                self._log_execution_pass(pass_index, total_passes, decisions)

                # Build preprocessing pipeline tailored to the profile
                self.log("Building preprocessor...")
                preprocessor = build_preprocessor(profile)
                # Choose candidate models to try based on the profile
                self.log("Selecting candidate models...")
                preferred_model = self._preferred_model_from_plan(plan)
                candidates = select_models(
                    profile,
                    seed=self.ctx.seed,
                    preferred_model=preferred_model,
                )
                candidate_names = [n for n, _ in candidates]
                if self.state.get("last_candidate_names") == candidate_names:
                    self.log("Candidate models unchanged.")
                else:
                    self.log(f"Candidate models: {candidate_names}")
                    self.state["last_candidate_names"] = candidate_names

                # Train candidate models and persist intermediate artefacts
                self.log("Training models...")

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        results = train_models(
                            df=df,
                            target=self.ctx.target,
                            preprocessor=preprocessor,
                            candidates=candidates,
                            seed=self.ctx.seed,
                            test_size=self.ctx.test_size,
                            output_dir=self.ctx.output_dir,
                            verbose=self.verbose,
                            is_classification=profile.get("is_classification", True),
                            apply_oversampling=profile.get("apply_oversampling", False)
                        )
                        break  # Exit retry loop if training succeeded
                    except Exception as e:
                        self.log(f"Training attempt {attempt + 1} failed with error: {e}")
                        if attempt + 1 == max_retries:
                            self.log("Max training attempts reached. Aborting run.")
                            return self.ctx.output_dir

                # Tune the best model if the plan requests it
                if "tune_hyperparameters" in plan:
                    tuning_skip_reason = self._tuning_skip_reason(
                        results,
                        is_classification=profile.get("is_classification", True),
                    )
                    if tuning_skip_reason:
                        self.log(f"Skipping hyperparameter tuning: {tuning_skip_reason}.")
                        profile.setdefault("notes", []).append(
                            f"Skipped hyperparameter tuning: {tuning_skip_reason}."
                        )
                    else:
                        self.log("Tuning best model hyperparameters...")
                        results = tune_best_model(
                            results,
                            seed=self.ctx.seed,
                            is_classification=profile.get("is_classification", True),
                            reduce_tuning_budget=profile.get("reduce_tuning_budget", False),
                        )
                        tuned = results.get("best", {}).get("tuned", False)
                        best_params = results.get("best", {}).get("best_params", {})
                        if tuned:
                            self.log(f"Tuning complete. Best params: {best_params}")
                        else:
                            self.log("Tuning skipped (no param grid for this model type).")

                # Evaluate the trained models and pick the best one
                eval_payload = self._evaluate_best_compat(results, profile)
                if "validate_with_cross_validation" in plan:
                    self.log("Validating top candidate models with cross-validation...")
                    cv_top_k = CV_TOP_K if profile["shape"]["rows"] >= SMALL_DATASET_ROWS else CV_TOP_K_SMALL
                    cv_payload = cross_validate_top_models(
                        df=df,
                        target=self.ctx.target,
                        training_payload=results,
                        seed=self.ctx.seed,
                        is_classification=profile.get("is_classification", True),
                        top_k=cv_top_k,
                    )
                    if cv_payload.get("enabled"):
                        self.log(
                            f"Cross-validation complete: {cv_payload.get('n_splits', 0)} folds "
                            f"across {len(cv_payload.get('models', []))} model(s)."
                        )
                    else:
                        self.log(f"Cross-validation skipped: {cv_payload.get('reason', 'unknown reason')}")
                else:
                    cv_payload = {
                        "enabled": False,
                        "reason": "Cross-validation was not requested by the plan.",
                        "n_splits": 0,
                        "models": [],
                        "warnings": [],
                    }
                eval_payload["cross_validation"] = cv_payload

                # Reflect on the evaluation in the context of the dataset profile
                reflection = reflect(
                    dataset_profile=profile,
                    evaluation=eval_payload["best_metrics"],
                    all_metrics=eval_payload["all_metrics"],
                    training_warnings=results.get("training_warnings", []),
                    cv_summary=cv_payload,
                )
                verdict = derive_run_verdict(profile, eval_payload, reflection)

                # Persist core run artefacts for later review
                save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"), profile)
                save_json(os.path.join(self.ctx.output_dir, "plan.json"), {"plan": plan})
                save_json(os.path.join(self.ctx.output_dir, "metrics.json"), eval_payload)
                save_json(os.path.join(self.ctx.output_dir, "reflection.json"), reflection)

                # Generate a human-readable markdown report summarising the run
                write_markdown_report(
                    out_path=os.path.join(self.ctx.output_dir, "report.md"),
                    ctx=self.ctx,
                    fingerprint=fp,
                    dataset_profile=profile,
                    plan=plan,
                    eval_payload=eval_payload,
                    reflection=reflection,
                )

                # Update the memory store with outcomes from this run
                # Only reliable runs are reused as successful priors; other runs are
                # stored as diagnostics so memory does not learn the wrong lesson.
                record = {
                    "last_seen": now_iso(),
                    "dataset": self.ctx.data_path,
                    "target": self.ctx.target,
                    "target_source": target_source,
                    "target_origin": target_origin,
                    "shape": profile["shape"],
                    "is_classification": profile.get("is_classification", True),
                    "verdict_label": verdict["label"],
                    "verdict_detail": verdict["detail"],
                    "reflection_status": reflection["status"],
                    "review_required": reflection.get("review_required", False),
                }
                if verdict["label"] == "Reliable result":
                    record["best_model"] = eval_payload["best_metrics"]["model"]
                    record["best_metrics"] = eval_payload["best_metrics"]
                else:
                    record["diagnostic_model"] = eval_payload["best_metrics"]["model"]
                    record["diagnostic_metrics"] = eval_payload["best_metrics"]
                if cv_payload.get("enabled"):
                    record["cross_validation"] = cv_payload
                self.memory.upsert_dataset_record(fp, record)

                # Invalid or baseline-beating failures should not be auto-reused later.
                best_model = eval_payload["best_metrics"]["model"]
                if verdict["label"] == "Invalid due to leakage risk":
                    self.log(f"Target '{self.ctx.target}' was flagged as invalid due to leakage risk — storing as failed.")
                    self.memory.add_failed_target(self.ctx.data_path, self.ctx.target)
                elif reflection["status"] == "needs_attention" and "Dummy" in best_model:
                    self.log(f"Target '{self.ctx.target}' produced no useful results — storing as failed.")
                    self.memory.add_failed_target(self.ctx.data_path, self.ctx.target)

                # Decide whether the agent should attempt to re-plan and re-run
                if not should_replan(reflection):
                    # No replan suggested — finish the run
                    break

                # If we've already replanned the allowed number of times, stop
                if self.state["replan_count"] >= self.ctx.max_replans:
                    self.log("Replan suggested, but max_replans reached. Stopping.")
                    break

                # Otherwise, increment replan counter and apply the replan strategy
                self.state["replan_count"] += 1
                self.log(f"Replanning attempt #{self.state['replan_count']}...")

                # apply_replan_strategy returns an updated (plan, profile) pair
                old_plan = list(plan)
                plan, profile = apply_replan_strategy(plan, profile, reflection)
                self._log_replan_diff(old_plan, plan, reflection)

                # If the replan signals the test split should be widened, apply it
                # (capped at 0.35 to keep enough training data)
                if profile.pop("increase_test_size", False):
                    new_test_size = min(self.ctx.test_size + 0.10, 0.35)
                    if new_test_size > self.ctx.test_size:
                        self.log(f"Widening test split: {self.ctx.test_size:.2f} → {new_test_size:.2f} to get a more representative held-out estimate.")
                        self.ctx.test_size = new_test_size

            if target_source == "inferred" and self._auto_target_is_unsuitable(
                eval_payload,
                verdict,
                profile.get("is_classification", True),
            ):
                failed_target = self.ctx.target
                self.log(
                    f"Auto target '{failed_target}' produced weak signal for auto-selection — storing as failed."
                )
                self.memory.add_failed_target(self.ctx.data_path, failed_target)
                auto_target_candidates = [candidate for candidate in auto_target_candidates if candidate != failed_target]

                if auto_target_candidates:
                    self.ctx.target = auto_target_candidates[0]
                    next_type = "classification" if is_classification_target(df[self.ctx.target]) else "regression"
                    self.log(f"Trying next inferred target: {self.ctx.target} (type: {next_type})")
                    self.state["replan_count"] = 0
                    self.state.pop("last_decision_summary", None)
                    self.state.pop("last_candidate_names", None)
                    continue

                self.log("No more strong auto-target candidates remain. Keeping the last run for review.")
            break

        # Final log and return the directory containing run outputs
        self._log_final_summary(eval_payload, verdict)
        return self.ctx.output_dir
