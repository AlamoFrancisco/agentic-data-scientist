# Setup Guide for CE888 Agentic Data Scientist

This guide will help you set up your development environment and get started with the assignment.

---

## Prerequisites

- **Python 3.9 or higher** (tested on 3.11)
- **Git** for version control
- **Text editor or IDE** (VS Code, PyCharm, Sublime, etc.)
- **Terminal/Command Line** access

---

## Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/AlamoFrancisco/agentic-data-scientist.git

# Navigate into the directory
cd agentic-data-scientist
```

---

## Step 2: Create a Virtual Environment

### On Windows:

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

### On macOS/Linux:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

---

## Step 3: Install Dependencies

```bash
# Upgrade pip first (recommended)
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

This will install:
- pandas, numpy, scipy (data/numerical)
- scikit-learn (machine learning)
- imbalanced-learn (SMOTE oversampling)
- matplotlib (visualization)
- pytest, pytest-cov (testing)

---

## Step 4: Verify Installation

```bash
# Test that the skeleton works
python run_agent.py --data data/example_dataset.csv --target auto
```

You should see output like:
```
[AgenticDataScientist] Loading dataset: data/example_dataset.csv
[AgenticDataScientist] Loaded 20 rows × 6 cols
[AgenticDataScientist] Inferred target: label
...
outputs/20250129_103045_a1b2c3d4
```

Check the `outputs/` directory for generated files:
- `report.md` - Human-readable summary
- `eda_summary.json` - Dataset profile
- `plan.json` - Execution plan
- `metrics.json` - Model metrics
- `reflection.json` - Agent reflection
- `confusion_matrix.png` - Visualization

---

## Step 5: Run Sanity Check

```bash
python tests/sanity_check.py
```

If everything is set up correctly, you'll see:
```
Running: python run_agent.py --data data/example_dataset.csv --target auto --quiet
Output dir: outputs/20250129_103045_a1b2c3d4
Sanity check passed.
```

---

## Step 6: Set Up Your Development Workflow

### Create a new branch for your work:

```bash
git checkout -b develop
```

### Make regular commits:

```bash
# After making changes
git add .
git commit -m "Implemented sophisticated planner logic"
git push origin develop
```

### IDE Setup (VS Code example):

1. Install Python extension
2. Select your virtual environment:
   - Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (Mac)
   - Type "Python: Select Interpreter"
   - Choose the one from `venv/`

3. Recommended extensions:
   - Python (Microsoft)
   - Pylance
   - Python Test Explorer
   - GitLens

---

### Manual release workflow

If you'd like to make the repository public at your own discretion, a manual GitHub Actions workflow is provided that will perform the change when you explicitly trigger it.

Steps to use it safely:

1. Create a personal access token (PAT) with **repo** scope and add it to the repository secrets as `GH_PAT` (Settings → Secrets and variables → Actions → New repository secret).
2. Push your code to the private repository as usual.
3. To make the repository public when you're ready:
   - Go to the **Actions** tab and open the workflow named **Make repository public (manual)**.
   - Click **Run workflow** and set the `confirm` input to `YES` (this prevents accidental runs).
   - Confirm and run; the workflow will use the `GH_PAT` secret to change the repository visibility and optionally create a release.

> Note: The workflow will only run when you manually trigger it and provide `confirm=YES`. It will not run automatically on a schedule.

**CI Note:** Continuous Integration currently runs on **Python 3.11** only (kept pinned for faster, more reliable runs); see `.github/workflows/ci.yml` if you want to restore a multi-version matrix.

---

## Step 7: Start Developing

### Priority order:

1. **Week 1-2:** Understand the skeleton code
   - Read through all files
   - Run the basic example
   - Trace execution flow
   - Identify TODOs

2. **Week 3-4:** Extend Planner and Reflector
   - Start with `agents/planner.py`
   - Then `agents/reflector.py`
   - Test each change

3. **Week 5-7:** Add advanced features
   - Choose 3+ features to implement
   - Test thoroughly

4. **Week 8:** Testing and documentation
   - Write comprehensive tests
   - Update docstrings
   - Update README

5. **Week 9:** Report and demo prep
   - Write technical report
   - Prepare demonstration
   - Practice timing

---

## Common Issues & Solutions

### Issue: `ModuleNotFoundError`

**Solution:**
```bash
# Make sure virtual environment is activated
# Check with: which python (should point to venv)

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Permission denied on script execution

**Solution:**
```bash
# On Windows, make sure you're in an activated venv
# On macOS/Linux:
chmod +x run_agent.py
```

### Issue: Outputs not being created

**Solution:**
```bash
# Create the directory manually
mkdir -p outputs
```

### Issue: Tests failing

**Solution:**
```bash
# Check Python version
python --version  # Should be 3.7+

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Run tests with verbose output
pytest -v tests/
```

---

## Development Tips

### 1. Use Version Control Effectively

```bash
# Commit after each feature
git add agents/planner.py
git commit -m "Added conditional planning for imbalanced datasets"

# Create feature branches
git checkout -b feature/hyperparameter-tuning
```

### 2. Test Incrementally

```bash
# Test small changes immediately
python run_agent.py --data data/example_dataset.csv --target auto

# Don't wait until everything is complete
```

### 3. Use Logging for Debugging

```python
# Add print statements liberally
print(f"DEBUG: plan = {plan}")
print(f"DEBUG: reflection issues = {reflection['issues']}")
```

### 4. Document as You Go

```python
# Write docstrings when you write the function
def my_new_function(x):
    """
    Brief description.
    
    Args:
        x: Description
    
    Returns:
        Description
    """
    pass
```

---

## Testing Your Code

### Run all tests:

```bash
pytest tests/
```

### Run with coverage:

```bash
pytest --cov=agents --cov=tools tests/
```

### Generate HTML coverage report:

```bash
pytest --cov=agents --cov=tools --cov-report=html tests/
# Open htmlcov/index.html in browser
```

---

## Adding New Datasets

1. Download dataset CSV
2. Place in `data/` directory (or provide download link if >10MB)
3. Document in `data/README.md`
4. Test your agent:

```bash
python run_agent.py --data data/your_dataset.csv --target target_column
```

---

## Getting Help

- **Forum:** Post on Moodle (no code sharing)
- **Lab Sessions:** Bring specific questions
- **Office Hours:** Book via [system]
- **Documentation:** Check docstrings and comments

---

## Useful Commands Reference

```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Run agent
python run_agent.py --data data/file.csv --target auto

# Run tests
pytest tests/

# Check code style
flake8 agents/ tools/ --max-line-length=100

# Format code (if black is installed)
black agents/ tools/

# Deactivate environment
deactivate
```

---

## Next Steps

1. ✅ Clone repository
2. ✅ Set up virtual environment
3. ✅ Install dependencies
4. ✅ Run basic example
5. ✅ Run sanity check
6. ✅ Read the assignment brief thoroughly
7. ✅ Understand the skeleton code
8. ✅ Start extending Planner and Reflector
9. ✅ Test continuously
10. ✅ Document as you go
11. ✅ Write technical report
12. ✅ Prepare demonstration
13. ✅ Practice timing

---

**You're all set! Start building your agentic data scientist!** 🚀

For assignment details, see the full brief on Moodle.
