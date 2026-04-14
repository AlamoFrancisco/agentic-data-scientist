# Orchestrator for an "agentic" offline data scientist pipeline.
# Handles dataset loading, profiling, planning, training, evaluation, reflection,
# and optional re-planning cycles. Designed primarily for classification tasks.
import os
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Agent components and tooling used by the orchestrator
from agents.planner import create_plan
from agents.reflector import reflect, should_replan, apply_replan_strategy
from agents.memory import JSONMemory
from tools.data_profiler import profile_dataset, infer_target_column, dataset_fingerprint, is_classification_target
from tools.modelling import build_preprocessor, cross_validate_top_models, select_models, train_models
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

    def load_data(self, path: str) -> pd.DataFrame:
        """Load a CSV into a pandas DataFrame and log its shape."""
        self.log(f"Loading dataset: {path}")
        df = pd.read_csv(path, na_values=["?", "NA", "N/A", "NULL", ""])
        self.log(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
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
        target_candidate_scores = None
        if target.strip().lower() == "auto":
            prev_hint = self.memory.get_dataset_record(
                "",
                dataset_name=self.ctx.data_path,
                require_reliable=True,
            )
            failed = self.memory.get_failed_targets(self.ctx.data_path)
            if prev_hint and prev_hint.get("target") and prev_hint["target"] not in failed:
                stored_target = prev_hint["target"]
                self.ctx.target = stored_target
                target_source = "memory"
                self.log(f"Using target from memory: '{stored_target}'")
            else:
                inferred, target_candidate_scores = infer_target_column(df, return_scores=True)
                # Skip targets that previously failed on this dataset
                failed = self.memory.get_failed_targets(self.ctx.data_path)
                if failed and inferred in failed:
                    self.log(f"Skipping previously failed target '{inferred}'. Trying next candidate.")
                    sorted_candidates = sorted(target_candidate_scores.items(), key=lambda x: x[1], reverse=True)
                    inferred = next((c for c, _ in sorted_candidates if c not in failed), inferred)
                if not inferred:
                    raise ValueError("Could not infer target column. Please provide --target <name>.")
                target_type = "classification" if is_classification_target(df[inferred]) else "regression"
                self.ctx.target = inferred
                target_source = "inferred"
                self.log(f"Inferred target: {inferred} (type: {target_type})")

        # Produce a dataset profile (EDA summary) and a fingerprint used for memory
        profile = profile_dataset(df, self.ctx.target, target_source=target_source, target_candidate_scores=target_candidate_scores)
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

        # Create an initial plan informed by the profile and optional memory hint
        plan = create_plan(profile, memory_hint=prev)
        self.log(f"Plan: {plan}")

        # Execution loop: trains and evaluates, then optionally replans and repeats
        while True:
            # Conditional execution based on plan
            if "consider_imbalance_strategy" in plan:
                self.log("Plan includes imbalance strategy — will use class weights.")
                profile["use_class_weights"] = True

            if "apply_regularization" in plan:
                self.log("Plan includes regularization — small dataset detected.")
                profile["use_regularization"] = True

            if "handle_severe_missing_data" in plan:
                self.log("Plan includes missing data handling — applying robust imputation.")
                profile["robust_imputation"] = True

            if "apply_target_encoding" in plan:
                self.log("Plan includes target encoding — high cardinality categorical detected.")
                profile["use_target_encoding"] = True

            if "apply_feature_engineering" in plan:
                self.log("Plan includes feature engineering — adding derived features.")
                profile["use_feature_engineering"] = True

            if "apply_robust_scaling" in plan:
                self.log("Plan includes robust scaling — scale mismatch detected.")
                profile["use_robust_scaling"] = True

            if "handle_outliers" in plan:
                self.log("Plan includes outlier handling.")
                profile["handle_outliers"] = True

            if "drop_near_constant_features" in plan:
                cols = profile.get("near_constant_cols", [])
                self.log(f"Plan includes dropping near-constant features: {cols}")

            if "drop_correlated_features" in plan:
                cols = profile.get("corr_cols_to_drop", [])
                self.log(f"Plan includes dropping correlated features: {cols}")
                profile["drop_high_corr"] = True

            if "drop_leaky_features" in plan:
                cols = profile.get("leaky_col_names", [])
                self.log(f"Plan includes dropping leaky features: {cols}")
                profile["drop_leaky"] = True

            if "use_simple_models_only" in plan:
                self.log("Plan includes simple models only — small dataset.")
                profile["simple_models_only"] = True

            if "use_ensemble_models" in plan:
                self.log("Plan includes ensemble models — large dataset.")
                profile["prefer_ensemble"] = True

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
            if preferred_model:
                self.log(f"Applying memory priority: {preferred_model}")
            self.log(f"Candidate models: {[n for n, _ in candidates]}")

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
                    )
                    break  # Exit retry loop if training succeeded
                except Exception as e:
                    self.log(f"Training attempt {attempt + 1} failed with error: {e}")
                    if attempt + 1 == max_retries:
                        self.log("Max training attempts reached. Aborting run.")
                        return self.ctx.output_dir

            # Evaluate the trained models and pick the best one
            eval_payload = evaluate_best(results, output_dir=self.ctx.output_dir, is_classification=profile.get("is_classification", True))
            if "validate_with_cross_validation" in plan:
                self.log("Validating top candidate models with cross-validation...")
                cv_top_k = 2 if profile["shape"]["rows"] >= 1000 else 3
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
                "shape": profile["shape"],
                "is_classification": profile.get("is_classification", True),
                "verdict_label": verdict["label"],
                "verdict_detail": verdict["detail"],
                "reflection_status": reflection["status"],
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
            plan, profile = apply_replan_strategy(plan, profile, reflection)

        # Final log and return the directory containing run outputs
        self.log(f"Done. Outputs saved to: {self.ctx.output_dir}")
        return self.ctx.output_dir
