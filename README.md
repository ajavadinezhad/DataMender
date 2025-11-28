# 🧹 DataMender

> **Smart data cleaning tool for large CSV/Parquet files using AI-powered rule discovery**

[![Status](https://img.shields.io/badge/status-complete-success?style=flat-square)](README.md)
[![Python](https://img.shields.io/badge/python-3.13+-blue?style=flat-square)](README.md)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](README.md)

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [LLM Setup (Optional)](#llm-setup-optional)
- [Usage Guide](#usage-guide)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Features

### Data Profiling & Analysis

- **Fast profiling** - Handles multi-GB datasets with hundreds of thousands of rows per second
- **Smart sampling** - Profile subset of rows for speed without losing accuracy
- **Comprehensive statistics** - Column types, nulls, ranges, uniqueness, and more

### Rule Discovery

- **Heuristic rules** - Instant rule discovery without any LLM required
- **AI-powered suggestions** - Leverage Groq (cloud) or Ollama (local) for intelligent rule generation
- **Batched LLM calls** - Process all columns in one efficient API call
- **Multi-mode execution** - Supports Heuristic-only, Groq, and Ollama configurations

### Human-in-the-Loop

- **Interactive Streamlit UI** - Beautiful, intuitive interface for rule validation
- **Accept/Reject rules** - Review and select rules before applying
- **Rule details** - See type, description, action, and severity for each rule

### Data Cleaning Engine

- **Vectorized operations** - Fast batch fixes using Polars
- **Multiple cleaning actions** - `clip_range`, `fill_null`, `abs_value`, `drop_rows`, `cross_column_check`, and more
- **Automatic chunking** - Handles multi-GB files with automatic chunked processing
- **Incremental cleaning** - Apply rules progressively without losing previous transformations
- **Progress tracking** - Real-time progress bars during cleaning operations

### Metrics & Validation

- **Before/after comparison** - Comprehensive metrics showing data quality improvements
- **Anomaly detection** - Identifies and tracks fixed issues automatically
- **Cumulative statistics** - Running totals across multiple cleaning passes
- **Cleaning logs** - Detailed log of all applied rules with cumulative statistics

### Export & Integration

- **Preview cleaned data** - See results before exporting
- **Export cleaned data** - Save as CSV or Parquet (supports chunked export for large files)
- **Export rules** - Save discovered rules as YAML or JSON for reuse

### Quality Assurance

- **Comprehensive testing** - Automated test suite covering all functionality
- **Error handling** - Robust handling of edge cases and invalid inputs
- **Performance validation** - Benchmarking and performance tracking

---

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv/bin/activate.fish` for fish shell

# Install dependencies
pip install -r requirements.txt

# Generate sample dataset (optional)
python src/generate_sample_data.py
```

### Generate Test Files

Generate test files of various sizes (25K to 20M records) for testing auto-chunking and large file handling:

```bash
# Generate test files in test_files/ directory
python generate_test_files.py
```

This will create CSV and Parquet files of increasing sizes:
- **Small**: 25K, 50K rows
- **Medium**: 100K, 250K rows
- **Large**: 500K, 1M rows
- **XLarge**: 2.5M, 5M rows
- **Huge**: 10M, 20M rows (maximum)

Files are saved to the `test_files/` directory. The script automatically:
- Creates both CSV and Parquet versions
- Adds intentional data quality issues for testing
- Limits maximum size to 20M records for safety

### Run Demo

```bash
# Run complete workflow demo
python demo_script.py sample_rides.csv 10000
```

### Run Tests

```bash
# Run comprehensive end-to-end tests
python tests/test_e2e.py

# Or run all tests
python tests/test_unit.py && python tests/test_integration.py && python tests/test_cli.py && python tests/test_e2e.py
```

---

## LLM Setup (Optional)

> **Heuristics work great without any LLM!** But for AI-powered suggestions, choose one:

### Option 1: Groq (Cloud - Recommended for Speed)

1. Get free API key at: [https://console.groq.com](https://console.groq.com)
2. Create `.env` file:
   ```bash
   GROQ_API_KEY=gsk_your_key_here
   ```
3. Select "groq" in the app - **super fast** (~5-10 seconds for all columns)

### Option 2: Ollama (Local - Unlimited & Private)

1. Install Docker (if not already installed)
2. Run Ollama container:
   ```bash
   docker run -d \
     --name ollama \
     -p 11434:11434 \
     -v ollama_data:/root/.ollama \
     ollama/ollama

   # Pull a model (3GB download)
   docker exec -it ollama ollama pull llama3.2
   ```
3. Select "ollama" in the app - runs locally (~15-30 seconds)

> **No LLM?** No problem! Heuristics discover rules instantly without any AI.

---

## Usage Guide

### Streamlit UI

#### Running the Application

```bash
# Activate venv first
source venv/bin/activate  # or venv/bin/activate.fish for fish

# Run the app
venv/bin/python -m streamlit run src/app.py
```

**Open:** http://localhost:8501

#### Workflow

**1. Configure** (sidebar)

- Choose LLM provider: `groq` (fast) or `ollama` (local)
- Toggle "Use LLM" checkbox (or use heuristics only)
- Set sample size (default 10000 rows, 0 = all rows)

**2. Load Data** (tab 1)

- Upload CSV/Parquet (up to 5GB supported)
- Click "Profile Dataset" - see statistics instantly
- Profiling uses Polars for speed

**3. Review Rules** (tab 2)

- See heuristic rules and AI rules
- Check boxes to accept/reject rules
- Each rule shows type, description, action, severity

**4. Clean Data** (tab 3)

- Click "Clean Data" to apply accepted rules
- View before/after comparison
- See cleaning metrics and statistics
- Preview cleaned data
- View detailed cleaning logs

**5. Export** (tab 4)

- Export accepted rules as YAML or JSON
- Export cleaned dataset as CSV or Parquet
- Ready to use in your data pipeline

### Environment Modes

DataMender supports 3 distinct runtime modes, verified across all configurations:

| Mode                     | Description                               | Configuration                                                           |
| ------------------------ | ----------------------------------------- | ----------------------------------------------------------------------- |
| **Heuristic only** | Rule discovery using heuristics only      | `export GROQ_API_KEY=''; unset DATAMENDER_TEST_OLLAMA`                |
| **Groq (Cloud)**   | Uses Groq LLM API for AI rule suggestions | `export GROQ_API_KEY=gsk_your_key_here; unset DATAMENDER_TEST_OLLAMA` |
| **Ollama (Local)** | Uses local Ollama model (via Docker)      | `export DATAMENDER_TEST_OLLAMA=1; export GROQ_API_KEY=''`             |

> **Tip:** If both Groq and Ollama are set, Groq takes priority and Ollama serves as fallback.

---

## Testing

### Test Organization

The test suite is organized into four comprehensive categories:

#### Unit Tests (`tests/test_unit.py`)

**Individual component tests:**

- Profiler initialization and column statistics
- Rule discovery initialization and heuristic rules
- Data cleaner initialization and action normalization
- Metrics initialization

#### Integration Tests (`tests/test_integration.py`)

**Complete workflow tests:**

- Complete workflow: Profile → Rules → Clean → Metrics
- Streamlit UI workflow simulation

#### CLI Tests (`tests/test_cli.py`)

**Command-line interface tests:**

- Profiler CLI execution
- Rule discovery CLI execution
- Demo script validation

#### Comprehensive E2E Tests (`tests/test_e2e.py`)

**Full test suite covering:**

- Data Profiler
- Rule Discovery
- Human Validation & Export
- Data Cleaning Engine
- Metrics & Re-Profiling
- Integration & Performance

### Running Tests

```bash
# Run tests individually
python tests/test_unit.py          # Unit tests (fast, isolated)
python tests/test_integration.py   # Integration tests (workflows)
python tests/test_cli.py           # CLI tests (command-line)
python tests/test_e2e.py           # Comprehensive E2E tests (full coverage)

# Run all tests sequentially
python tests/test_unit.py && \
python tests/test_integration.py && \
python tests/test_cli.py && \
python tests/test_e2e.py

# Or using pytest (if installed)
pytest tests/
```

**Test Suite Status:** All tests passing ✅

### Test Coverage

**Total Tests: 98** (All passing ✅)

#### Test Categories Breakdown

- **Unit Tests**: 23 tests
- **E2E Tests**: 73 tests
- **Integration Tests**: 2 tests
- **CLI Tests**: 1 test (3 functions, but 1 runs standalone)

#### Code Coverage

**Overall Coverage: 79%** (Core library modules, excluding Streamlit UI)

| Module                            | Statements | Coverage       |
| --------------------------------- | ---------- | -------------- |
| **profiler.py**             | 66         | **89%**  |
| **rule_discovery.py**       | 116        | **89%**  |
| **data_cleaner.py**         | 275        | **80%**  |
| **metrics.py**              | 138        | **74%**  |
| **generate_sample_data.py** | 62         | **74%**  |
| **llm_client.py**           | 47         | **62%**  |
| **__init__.py**       | 1          | **100%** |

> **Note:** Streamlit UI (`app.py`) is not included in coverage metrics as UI components are tested manually through interactive use.

---

## Project Structure

```
DataMender/
├── src/
│   ├── profiler.py              # Fast data profiling with Polars
│   ├── rule_discovery.py        # Heuristic + LLM rule generation
│   ├── llm_client.py            # LLM abstraction (Groq/Ollama)
│   ├── data_cleaner.py          # Vectorized data cleaning engine
│   ├── metrics.py               # Before/after comparison metrics
│   ├── app.py                   # Streamlit UI with human-in-the-loop
│   └── generate_sample_data.py  # Generate test dataset
├── tests/                       # Test suite directory
│   ├── __init__.py              # Test package initialization
│   ├── test_unit.py             # Unit tests
│   ├── test_integration.py      # Integration tests
│   ├── test_cli.py              # CLI tests
│   └── test_e2e.py              # Comprehensive E2E tests
├── .streamlit/
│   └── config.toml              # Streamlit config (5GB upload limit)
├── demo_script.py               # Complete workflow demo script
├── requirements.txt             # Python dependencies
├── .env                         # API keys (gitignored)
└── README.md                    # This file
```

---

## Tech Stack

| Technology            | Purpose                                                |
| --------------------- | ------------------------------------------------------ |
| **Polars**      | Fast DataFrame operations (10-100x faster than pandas) |
| **Streamlit**   | Interactive web UI                                     |
| **Groq**        | Cloud LLM API (free tier, super fast)                  |
| **Ollama**      | Local LLM server (Docker-based)                        |
| **Python 3.13** | Modern Python with venv                                |
| **PyYAML**      | Rule export and configuration                          |
| **python-dotenv** | Environment variable management for API keys          |
| **psutil**      | Performance monitoring                                 |

---

## Project Status

**✅ Complete (Weeks 1-7)**

All core features have been fully implemented:

- ✅ Data Profiler - Fast CSV/Parquet analysis
- ✅ Rule Discovery - Heuristic + LLM rule generation
- ✅ Human Validation - Interactive Streamlit UI
- ✅ Data Cleaning Engine - Vectorized data transformation
- ✅ Re-Profiling & Metrics - Before/after comparison
- ✅ Preview & Export - Cleaned data preview and export
- ✅ Comprehensive Testing - Full test suite coverage

---

<div align="center">

**Made with ❤️ for data quality**

[⬆ Back to Top](#-datamender)

</div>
