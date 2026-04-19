CE888 – Data Science and Decision MakingCoursework (82%)
Design and Evaluation of an Offline Agentic Data Scientist
School of Computer Science and Electronic Engineering
University of Essex
Jan-2026
Coursework Overview
This coursework requires you to design, implement, and evaluate an Agentic AI system
capable of autonomously performing data science tasks on unseen datasets. The focus is on
autonomy, planning, reflection, and system-level reasoning rather than maximising predictive
accuracy. The coursework contributes 82% of the module mark and is designed for approx-
imately 70–80 hours of independent study.
Learning Outcomes Assessed
By completing this coursework, you will demonstrate your ability to:
•design autonomous AI systems using agentic principles;
•apply data science techniques in a principled, data-aware manner;
•evaluate and reflect on AI system behaviour;
•communicate technical decisions clearly and responsibly.
1 Assessment Structure and Deliverables
All submissions must be made via FASER.
Deliverable Weight Submission Deadline
Data Exploration & Planning 20% FASER + Lab Demo 16 Feb 2026, 13:59:59
Final Project Code 31% FASER 21 Apr 2026, 13:59:59
Final Project Demonstration 31% FASER 21 Apr 2026, 13:59:59
2 Provided Materials
Please click here to GitHub template repository (Agentic Data Scientist skeleton), a small demo
dataset for local testing, and clear constraints on allowed tools. You must base your work on
the provided template. Please read the README file carefully before starting.
3 Constraints and Rules
Allowed:
•Python; pandas, NumPy, scikit-learn, matplotlib;
•open-source libraries only; fully offline execution.
Not Allowed:
•paid APIs; AutoML platforms; cloud-based ML services;
•copying full solutions from the internet.
1
CE888 2025–26
4 Deliverable 1: Data Exploration & Planning (20%)
Weight: 20%, Submission: FASER + in-lab demonstration, Deadline: 16 Feb 2026, 13:59:59
4.1 Purpose
To demonstrate dataset understanding, exploratory data analysis, and a clear plan for your
agentic system.
4.2 What you must submit to FASER
Upload ONE ZIP file named:
“CE888 DataExploration YourRegistrationNumber.zip”
ZIP file contents (mandatory)
4.3 File details
1. “eda.ipynb” (MANDATORY): Your notebook must include: dataset loading, exploratory
data analysis (plots + statistics), identification of: feature types, missing values, imbalance,
potential modelling challenges, brief data cleaning (where required). The notebook must run
top-to-bottom without errors.
2. “README.md” (MANDATORY): A short markdown file (1-2 pages) explaining: key find-
ings from EDA, dataset challenges, your proposed agentic plan for the final project: what
signals the agent will extract, how decisions will be made, what will trigger reflection and
re-planning.
4.4 In-lab demo (MANDATORY)
You must demonstrate “eda.ipynb” to a GLA or the module leader, explain your findings and
proposed plan. Important: If you do not submit to FASER and complete the lab demo, this
deliverable will receive 0 marks.
Dr Haider Raza 2 CSEE, University of Essex
CE888 2025–26
5 Deliverable 2: Final Project Code (31%)
Weight: 31%, Submission: FASER, Deadline: 21 Apr 2026, 13:59:59
5.1 Purpose
To assess the implementation quality of your Offline Agentic Data Scientist.
5.2 What you must submit to FASER
Upload ONE ZIP file named: “CE888 FinalCode YourRegistrationNumber.zip”
ZIP file contents (mandatory)
Your ZIP must contain a complete runnable project, based on the provided GitHub template:
Figure 1: Enter Caption
5.3 Code expectations
Your system must: run end-to-end on unseen datasets and should include:
1. autonomous planning
2. conditional model/metric selection
3. reflection and re-planning
4. persistent memory
5. produce outputs automatically (reports, metrics, plots)
5.4 README.md (MANDATORY)
Must clearly explain: system architecture, how to run the code, how planning, reflection, and
memory are implemented. Finally tip, poor documentation will be penalised.
Dr Haider Raza 3 CSEE, University of Essex
CE888 2025–26
6 Deliverable 3: Final Project Demonstration
Weight: 31%, Submission: FASER, Deadline: 21 Apr 2026, 13:59:59
6.1 Purpose
To assess understanding, reasoning, and coherence between your code and explanations.
6.2 What you must submit to FASER
Upload ONE ZIP file named: “CE888 FinalDemo YourRegistrationNumber.zip”
ZIP file contents (mandatory)
Your ZIP must contain a complete runnable project, based on the provided GitHub template:
6.3 File details
1. “demo.mp4” (MANDATORY): 8–10 minutes and must show: agent running on an unseen
dataset, planning decisions, reflection and (if applicable) re-planning.
2. “slides.pdf” (MANDATORY): Slides used in the video, covering: problem definition, agent
architecture, key decisions, results and limitations.
7 Marking Philosophy
Marks are awarded primarily for autonomy, reasoning, planning depth, reflection quality, system
design, and ethical awareness. Raw predictive accuracy carries low weight.
Final Advice
This coursework rewards students who think like AI system designers, not Kaggle competi-
tors. We assess your agent’s behaviour, not your model’s accuracy.
Time Management Tips
•Start early: 80 hours is accurate, not an overestimate
•Commit to GitHub regularly (after each feature)
•Test continuously: don’t wait until the end
•Ask questions early: use office hours and labs
•Document as you go: don’t leave it until the end