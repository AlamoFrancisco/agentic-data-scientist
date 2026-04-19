# Prompt for Generating the Presentation

*Copy everything below this line and paste it into Gemini or ChatGPT to generate your presentation slides and script.*

***

**System Prompt:**
You are an expert presentation designer and data science communicator. I need to create an 8 to 10-minute presentation (Deliverable 3) for my university coursework. The project is an **Offline Agentic Data Scientist**. 

Please generate a slide-by-slide outline (around 8 slides) and a suggested spoken script for each slide. The presentation must focus on system architecture, autonomous decision-making, and scientific reasoning rather than just chasing raw predictive accuracy. 

Slide 5 must be a placeholder for a 2.5-to-3 minute Live Video Demonstration of the agent running in the terminal.

Here is all the context and raw data you need to build the presentation:

## 1. Project Overview
- **Author:** Francisco Antonio Alamo Rios
- **Module:** CE888 (Data Science and Decision Making)
- **Concept:** An autonomous, offline AI system that executes end-to-end data science workflows (classification and regression) without Large Language Models. It uses rule-based reasoning, heuristics, and persistent memory.

## 2. System Architecture (Sense-Plan-Act-Reflect)
- **Profiler (Sense):** Extracts signals like leakage, scale mismatch, imbalance, and dirty data.
- **Planner (Plan):** Uses conditional logic to build a targeted execution plan (e.g., if highly imbalanced -> apply SMOTE; if massive dataset -> reduce tuning budget).
- **Executor & Modelling (Act):** Builds the preprocessing pipeline dynamically and trains candidate models.
- **Reflector (Reflect):** Analyses performance. Uses statistical significance (paired t-tests) to avoid picking complex models if a simple one is just as good. Triggers replanning if performance is weak.
- **Memory:** Remembers successful models per dataset and tracks "failed targets" to avoid repeating mistakes.

## 3. Key Scientific Design Decisions
- **Leakage-Aware Verdicts:** Uses Mutual Information (MI) to detect target proxies. It doesn't just silently drop soft proxies; it flags the run with a "Use with caution" verdict, forcing human review. (Also implemented a custom fix for continuous regression targets so they don't break MI calculations).
- **Autonomous Target Fallback:** If the user asks the agent to "auto-detect" the target, and it picks a column that turns out to have zero predictable signal, the Reflector aborts the run, saves the target to a "failed" list in memory, and automatically pivots to the next best target candidate.
- **Statistical Rigor:** Evaluates cross-validation consistency. If the held-out split differs wildly from the CV mean, it triggers a replan to widen the test split.

- **Outlier-Aware Regression Reflection:** If regression performance (R²) is low and the profiler detected outliers, the Reflector explicitly flags this issue and suggests applying robust scaling or using outlier-insensitive models.

## 4. The "Test Track" (The 5 Evaluation Datasets)
The agent was tested on 5 distinct datasets to prove its adaptability:

1. **Titanic (891 rows, Classification):**
   - *Challenge:* Hard data leakage and missing data.
   - *Agent Action:* Detected the `alive` column as a 100% proxy for `survived` and automatically dropped it. Dropped sensitive demographics (`sex`, `age`) for fairness. Re-planned to widen the test split when CV variance was high.
   - *Result:* LogisticRegression (bal_acc=0.825).

2. **Digits (1797 rows, Classification):**
   - *Challenge:* High dimensionality with useless features.
   - *Agent Action:* Detected 14 border pixels as "near-constant" (≥95% zeros) and dynamically pruned them from the pipeline.
   - *Result:* RandomForest (bal_acc=0.972).

3. **Sales (30,000 rows, Regression):**
   - *Challenge:* Massive workload and high-cardinality strings.
   - *Agent Action:* Applied `TargetEncoder` for product names (899 unique values). Scaled down the hyperparameter tuning budget automatically due to the 30k row size. Flagged `final_price` as a soft leakage proxy for `revenue`.
   - *Result:* RandomForestRegressor.

4. **Telco Churn (7043 rows, Classification):**
   - *Challenge:* Dirty data and class imbalance.
   - *Agent Action:* The `TotalCharges` column contained blank spaces, making Pandas load it as a string. The agent auto-coerced it to numeric, recovering the data. Detected the massive scale range and applied `RobustScaler`. Applied class weights for the churn imbalance.
   - *Result:* LogisticRegression (bal_acc=0.723).

5. **WineQuality (1700 rows, Regression):**
   - *Challenge:* Standard baseline regression.
   - *Agent Action:* Executed a clean, standard regression pipeline without triggering emergency safety rails.

## 5. Ethics & Limitations
- **Ethics:** The agent actively scans for sensitive keywords (age, gender, race) and drops them to mitigate direct algorithmic bias. It also calculates a Demographic Parity proxy to audit the final model's predictions across protected groups.
- **Limitations:** The leakage detector handles individual features perfectly, but it cannot currently detect complex multiplicative leakage (e.g., Feature A * Feature B = Target).

***
**Output Requirements:**
- Slide titles and bullet points.
- A conversational, professional spoken script for each slide.
- A specific suggested script for the Live Demo (Slide 5), imagining I am demonstrating the agent running on the **Titanic** dataset or the **Telco Churn** dataset.