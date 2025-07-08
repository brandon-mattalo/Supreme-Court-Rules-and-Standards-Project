# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based legal research application that analyzes Supreme Court of Canada (SCC) decisions to extract and compare legal tests. The application uses Streamlit for the web interface and integrates with Google's Gemini AI models for natural language processing. The application performs: (1) extraction of main legal tests from SCC decisions using AI; (2) efficient pairwise comparisons using Bradley-Terry linked block design; and (3) statistical analysis to determine whether the Supreme Court of Canada has moved towards more "standard-like" tests as opposed to "rule-like" tests.

## Version 2.0 Updates (2025-07-08)

### Major UI/UX Improvements
- **Experiment numbering**: All experiments now display with unique numbers (#1, #2, etc.)
- **Simplified navigation**: Removed lazy loading, experiments display by default in sidebar
- **Enhanced cost tracking**: Comprehensive cost estimates with extraction/comparison breakdowns
- **Improved AI configuration**: Better tooltips and Top K defaults for legal analysis
- **Performance optimization**: Extensive caching and connection pooling implemented

### Bradley-Terry Optimization
- **Linked block design**: Uses 15-case blocks (12 core + 3 bridge cases) instead of full pairwise comparisons
- **Massive efficiency gains**: 99.6% reduction in required comparisons for large datasets (4,835 cases: 42K vs 11.7M comparisons)
- **Centralized calculation**: `calculate_bradley_terry_comparisons()` function ensures consistency across application
- **Realistic cost estimates**: Based on actual comparison requirements, not theoretical maximums

## Key Commands

### Running the Application
```bash
streamlit run app.py
```

### Bulk Processing
```bash
python bulk_processor.py --api_key YOUR_API_KEY --model_name gemini-2.5-pro
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Architecture Overview

### Core Components

- **app.py**: Main application entry point with settings management and navigation  
- **pages/dashboard.py**: Core dashboard interface with experiment management, data loading, and analysis features
- **pages/experiment_execution.py**: Experiment execution interface for running extractions and comparisons
- **bulk_processor.py**: Command-line tool for batch processing cases without the web interface
- **config.py**: Configuration management including API keys, model settings, and database path
- **schemas.py**: Pydantic models for data validation (ExtractedLegalTest, LegalTestComparison)
- **utils/case_management.py**: Database operations and case management utilities

### Database Schema

The application uses SQLite/PostgreSQL with experiment-based architecture (v2.0):
- `v2_cases`: Stores SCC case metadata and full text (fresh v2 database)
- `v2_experiments`: Stores experiment configurations and metadata
- `v2_experiment_runs`: Stores execution results and statistics for each experiment run
- `v2_experiment_extractions`: Stores legal test extractions per experiment per case
- `v2_experiment_comparisons`: Stores pairwise comparisons per experiment between extractions
- `v2_experiment_results`: Stores analysis results including Bradley-Terry statistics

Key architectural change: Extractions and comparisons are now stored per experiment, allowing different experiments to have different extractions from the same cases.

### Data Flow

1. **Data Loading**: Parquet files are loaded and cross-referenced with Excel database for metadata
2. **Extraction**: AI models extract legal tests from case text using structured prompts
3. **Validation**: Human validators can review, edit, or mark extractions as accurate/inaccurate
4. **Pairwise Comparisons**: Users compare legal tests to determine which are more "rule-like"
5. **Analysis**: Bradley-Terry scoring and temporal analysis using real comparison data

### Key Features

- **Gemini Integration**: Uses Google's latest Gemini models (2.5 Pro/Flash) with structured JSON output
- **Persistent API Storage**: API keys are saved locally and persist across sessions
- **Accurate Cost Estimation**: Real-time cost calculations based on actual case lengths and per-million token pricing
- **AI Configuration Optimization**: Enhanced parameter tooltips, Top K slider (default: 40), and legal-specific guidance
- **Enhanced Validation UI**: Table-based validation interface with inline editing and action buttons
- **Efficient Comparison System**: Bradley-Terry linked block design reduces comparison requirements by 99%+
- **Statistical Analysis**: Connected graph validation, temporal trend analysis, and significance testing
- **Bulk Processing**: Command-line interface for processing multiple cases
- **Performance Optimization**: Comprehensive caching, connection pooling, and lazy loading
- **Experiment Management**: Numbered experiments with detailed progress and cost tracking

## Configuration

- **Database**: v2.0 uses separate database (`scc_cases_v2.db`) to avoid corrupting v1 data
- **API Keys**: Stored persistently in `.api_key.json` file (excluded from git via .gitignore)
- **AI Models**: Defaults to `gemini-2.5-pro`, with corrected pricing per million tokens:
  - gemini-2.5-flash: $0.30 input / $2.50 output per million tokens
  - gemini-2.5-pro: $1.25 input / $10.0 output per million tokens
- **AI Parameters**: Optimized defaults for legal analysis (Top K: 40, Temperature: 0.0)

## Performance Optimizations

### Database Caching
- `@st.cache_data(ttl=30)` for frequently changing data (database counts, selected cases)
- `@st.cache_data(ttl=60)` for moderately changing data (experiments, case summaries)
- `@st.cache_data(ttl=120)` for relatively static data (available cases)
- `@st.cache_resource` for database connections and engines

### UI Optimizations
- Removed lazy loading barriers for better user experience
- Page config optimization in app.py
- Cache invalidation after data modifications
- Connection pooling for PostgreSQL

## Bradley-Terry Implementation

### Block Design Parameters
- **Block size**: 15 cases (12 core + 3 bridge cases)
- **Comparisons per block**: 105 pairwise comparisons
- **Strategy**: Full pairwise for ≤15 cases, linked blocks for >15 cases
- **Function**: `calculate_bradley_terry_comparisons(n_cases)` in `utils/case_management.py`

### Efficiency Examples
- 10 cases: 45 comparisons (unchanged)
- 100 cases: 945 comparisons (vs 4,950 full pairwise - 81% reduction)
- 4,835 cases: 42,315 comparisons (vs 11.7M full pairwise - 99.6% reduction)

## Important Notes

- The application expects Excel metadata file at specific hardcoded path
- All AI extractions go through a validation workflow before being marked as accurate
- The system supports both individual case processing and bulk batch processing
- Bradley-Terry analysis uses efficient linked block design for scalable comparisons
- All cost estimates are based on real case lengths and accurate token pricing
- Experiment numbers are unique identifiers displayed throughout the interface