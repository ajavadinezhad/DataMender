# DataMender

Smart data cleaning tool for large CSV/Parquet files using AI-powered rule discovery.

## Features

- ⚡ **Fast profiling** of large datasets (CSV/Parquet, up to 5GB)
- 🤖 **AI-powered rule discovery** (Groq cloud or Ollama local)
- 🧠 **Heuristic rules** - works without LLM (15-20 rules found instantly)
- 👤 **Human-in-the-loop** validation UI with Streamlit
- 🚀 **Batched LLM calls** - all columns processed in one shot (~10 seconds)
- 📊 **Smart sampling** - profile subset of rows for speed

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `source venv/bin/activate.fish` for fish shell

# Install dependencies
pip install -r requirements.txt

# Generate sample dataset (optional)
python src/generate_sample_data.py

# Test CLI (optional)
python test_cli.py
```

## LLM Setup (Optional)

**Heuristics work great without any LLM!** But for AI-powered suggestions, choose one:

### Option 1: Groq (Cloud - Recommended for Speed)

1. Get free API key at: https://console.groq.com
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

**No LLM?** No problem! Heuristics discover 15-20 rules instantly without any AI.

## Usage

### Streamlit UI

```bash
# Activate venv first
source venv/bin/activate  # or venv/bin/activate.fish for fish

# Run the app
venv/bin/python -m streamlit run src/app.py
```

**Open:** http://localhost:8501

**Workflow:**
1. **Configure** (sidebar):
   - Choose LLM provider: `groq` (fast) or `ollama` (local)
   - Toggle "Use LLM" checkbox (or use heuristics only)
   - Set sample size (default 10000 rows, 0 = all rows)

2. **Load Data** (tab 1):
   - Upload CSV/Parquet (up to 5GB supported)
   - Click "Profile Dataset" - see statistics instantly
   - Profiling uses Polars for speed ⚡

3. **Review Rules** (tab 2):
   - See heuristic rules (⚡ Fast) and AI rules (🤖 AI)
   - Check boxes to accept/reject rules
   - Each rule shows type, description, action, severity

4. **Export Rules** (tab 3):
   - Export accepted rules as YAML or JSON
   - Ready to use in your data pipeline

### CLI Testing

```bash
python test_cli.py
```

This will test the profiler and rule discovery on the sample dataset.

## Project Structure

```
DataMender/
├── src/
│   ├── profiler.py          # Fast data profiling with Polars
│   ├── rule_discovery.py    # Heuristic + LLM rule generation
│   ├── llm_client.py        # LLM abstraction (Groq/Ollama)
│   ├── app.py              # Streamlit UI with human-in-the-loop
│   └── generate_sample_data.py  # Generate test dataset
├── .streamlit/
│   └── config.toml         # Streamlit config (5GB upload limit)
├── requirements.txt        # Python dependencies
├── .env                   # API keys (gitignored)
└── README.md
```

## Tech Stack

- **Polars**: Fast DataFrame operations (10-100x faster than pandas)
- **Streamlit**: Interactive web UI
- **Groq**: Cloud LLM API (free tier, super fast)
- **Ollama**: Local LLM server (Docker-based)
- **Python 3.13**: Modern Python with venv

