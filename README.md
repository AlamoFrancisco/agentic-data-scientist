# CE888 Agentic Data Scientist - Student Template

**Name** Francisco Antontio Alamo Rios
**Assignment:** Offline Agentic AI for Data Science  
**Module:** CE888  
**Academic Year:** 2025/2026

---

## Overview

This repository contains the skeleton code for building an **Offline Agentic Data Scientist** - an autonomous agent that performs end-to-end classification tasks without relying on Large Language Models.

Your agent will use **rule-based reasoning, heuristics, and meta-learning** to autonomously:
- Profile datasets
- Plan execution workflows
- Train and evaluate models
- Reflect on results
- Learn from experience

---

## Quick Start

### 1. Clone this repository

```bash
git clone https://github.com/sagihaider/ce888-agentic-data-scientist.git
cd ce888-agentic-data-scientist
```

### 2. Set up your environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Test the skeleton

```bash
python run_agent.py --data data/example_dataset.csv --target auto
```

You should see the agent run through the basic pipeline and generate outputs in `outputs/[timestamp]/`

---

## Project Structure

```
ce888-agentic-data-scientist/
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── agentic_data_scientist.py      # Core agent (Executor) - extend this
├── run_agent.py                   # Entry point - students run this
│
├── agents/                        # Agent components
│   ├── __init__.py
│   ├── planner.py                 # TODO: EXTEND THIS
│   ├── reflector.py               # TODO: EXTEND THIS
│   └── memory.py                  # Basic implementation - can extend
│
├── tools/                         # Data science tools
│   ├── __init__.py
│   ├── data_profiler.py          # Provided - can extend
│   ├── modelling.py              # Provided - can extend
│   └── evaluation.py             # Provided - can extend
│
├── data/                          # Datasets
│   ├── README.md                 # Dataset documentation template
│   └── example_dataset.csv       # Small demo dataset
│
├── outputs/                       # Generated outputs (gitignored)
│   └── .gitkeep
│
├── report/                        # Your technical report
│   ├── README.md                 # Report guidelines
│   └── REPORT.md                 # TODO: WRITE YOUR REPORT HERE
│
└── tests/                         # Test suite
    ├── __init__.py
    └── sanity_check.py           # Basic sanity test
```

---

## What You Need To Do

### Core Tasks (Mandatory)

1. **Extend the Planner Agent** (`agents/planner.py`)
   - Implement sophisticated planning logic based on dataset characteristics
   - Create different plan templates for different scenarios
   - Use memory hints to guide planning
   - Handle edge cases (small datasets, high imbalance, etc.)

2. **Extend the Reflector Agent** (`agents/reflector.py`)
   - Implement deep performance analysis
   - Add statistical significance testing
   - Generate actionable improvement suggestions
   - Implement smart replanning decisions

3. **Enhance the Executor** (`agentic_data_scientist.py`)
   - Add robust error handling
   - Implement retry logic
   - Add detailed logging
   - Support conditional execution based on plans

4. **Improve the Memory System** (`agents/memory.py`)
   - Add similarity-based retrieval
   - Implement richer experience storage
   - Enable learning from past executions

### Evaluation & Reporting

- Test on **at least 3 diverse classification datasets**
- Write a **3000-4000 word technical report** in `report/REPORT.md`
- Create comprehensive tests with >60% coverage
- Document all datasets in `data/README.md`

---

## Running the Agent

### Basic Usage

```bash
python run_agent.py --data data/example_dataset.csv --target auto
```

### Custom Parameters

```bash
python run_agent.py \
    --data data/your_dataset.csv \
    --target target_column_name \
    --output_root my_outputs \
    --seed 42 \
    --test_size 0.2 \
    --max_replans 2
```

### Arguments

- `--data`: Path to CSV dataset (required)
- `--target`: Target column name or 'auto' for automatic detection (required)
- `--output_root`: Output directory (default: 'outputs')
- `--seed`: Random seed for reproducibility (default: 42)
- `--test_size`: Test set fraction (default: 0.2)
- `--max_replans`: Maximum replanning attempts (default: 1)
- `--quiet`: Reduce logging output

---

## Testing

Run the sanity check:

```bash
python tests/sanity_check.py
```

Run all tests (once you add them):

```bash
pytest tests/
```

With coverage:

```bash
pytest --cov=agents --cov=tools --cov-report=html tests/
```

---

## Expected Output

After running the agent, check `outputs/[timestamp]/` for:

- `report.md` - Human-readable summary report
- `eda_summary.json` - Dataset profile and characteristics
- `plan.json` - Generated execution plan
- `metrics.json` - Model performance metrics
- `reflection.json` - Agent's self-assessment and suggestions
- `confusion_matrix.png` - Confusion matrix visualization

---

## Development Workflow

1. **Week 1-2:** Understand the skeleton, run basic examples
2. **Week 3-4:** Extend Planner and Reflector with sophisticated logic
3. **Week 5-7:** Implement 3+ advanced features
4. **Week 8:** Add comprehensive tests and documentation
5. **Week 9:** Write technical report and prepare demo

---

## Key Files to Modify

### High Priority (Must Extend)
- `agents/planner.py` - Your planning logic
- `agents/reflector.py` - Your reflection logic
- `agentic_data_scientist.py` - Enhanced executor

### Medium Priority (Should Extend)
- `agents/memory.py` - Richer memory system
- `tools/modelling.py` - Additional models or strategies
- `tests/test_*.py` - Your test cases

### Low Priority (Optional Extensions)
- `tools/data_profiler.py` - Additional profiling features
- `tools/evaluation.py` - More evaluation metrics
- New tool files for advanced features

---

## Submission Checklist

Before submitting, ensure:

- [ ] All code runs without errors
- [ ] README.md updated with your modifications
- [ ] requirements.txt includes any new dependencies
- [ ] At least 3 test datasets documented in `data/README.md`
- [ ] Technical report completed (3000-4000 words)
- [ ] Test coverage >60%
- [ ] All core components extended significantly
- [ ] At least 3 advanced features implemented
- [ ] GitHub repository is clean and organised

---

## Getting Help

- **Lectures:** All the lectures and labs are important. Each session will enhance your knowledge. 
- **Office Hours:** 9-10 am (Monday and Friday)
- **Forum:** Moodle discussion board - for clarifications (no code sharing)

---

## Important Deadlines

- **Data Exploration**. Deadline:  16-Feb-2026 13:59:59
- **Final Project Demonstration**. Deadline:  21-Apr-2026 13:59:59
- **Final Project Code**. Deadline:  21-Apr-2026 13:59:59

---

## Academic Integrity

- This is **individual work** - no code sharing with classmates
- You may use standard libraries and consult documentation
- AI assistance is allowed but must be disclosed in your report
- Cite any code adapted from external sources
- Plagiarism will result in severe penalties

---

## Resources

- **Scikit-learn:** https://scikit-learn.org/stable/
- **Pandas:** https://pandas.pydata.org/docs/
- **Assignment Brief:** See Moodle for full details
- **Dataset Sources:**
  - Kaggle: https://www.kaggle.com/datasets
  - UCI ML Repository: https://archive.ics.uci.edu/ml
  - OpenML: https://www.openml.org/

---

## Contact

**Module Leader:** Dr Haider Raza  

**Email:** use my essex email address

**Office Hours:** Monday and Friday 9-10 AM. 

---

## Release & License

- **License:** This project is released under the **MIT License** by default. See `LICENSE` for full terms. If you'd prefer a different license (e.g., Apache-2.0, CC-BY), replace `LICENSE` accordingly before publishing.

**Good luck with your assignment! Build something you're proud of!** 🚀
