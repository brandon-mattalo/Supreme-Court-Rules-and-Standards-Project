# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based legal research application that analyzes Supreme Court of Canada (SCC) decisions to extract and compare legal tests. The application uses Streamlit for the web interface and integrates with Google's Gemini AI models for natural language processing. The application will do two things: (1) extract the main legal test and/or tests from SCC decisions; and (2) after that is complete, do a pairwise comparison using the Bradley-Taylor model. After that it will display the results, with statistical regression and/or correlation analyses to determine whether the Supreme Court of Canada has moved towards more "standard-like" tests as opposed to "rule-like" tests.

## Key Commands

### Running the Application
```bash
streamlit run main_app.py
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

- **main_app.py**: Streamlit web application with data loading, extraction, validation, and analysis features
- **bulk_processor.py**: Command-line tool for batch processing cases without the web interface
- **config.py**: Configuration management including API keys, model settings, and database path
- **schemas.py**: Pydantic models for data validation (ExtractedLegalTest, LegalTestComparison)

### Database Schema

The application uses SQLite with three main tables:
- `cases`: Stores SCC case metadata and full text
- `legal_tests`: Stores extracted legal tests with validation status and Bradley-Terry scores
- `legal_test_comparisons`: Stores pairwise comparisons between legal tests for Bradley-Terry analysis

### Data Flow

1. **Data Loading**: Parquet files are loaded and cross-referenced with Excel database for metadata
2. **Extraction**: AI models extract legal tests from case text using structured prompts
3. **Validation**: Human validators can review, edit, or mark extractions as accurate/inaccurate
4. **Pairwise Comparisons**: Users compare legal tests to determine which are more "rule-like"
5. **Analysis**: Bradley-Terry scoring and temporal analysis using real comparison data

### Key Features

- **Gemini Integration**: Uses Google's latest Gemini models (2.5 Pro/Flash) with structured JSON output
- **Persistent API Storage**: API keys are saved locally and persist across sessions
- **Cost Estimation**: Calculates API costs with current 2025 pricing, including high-volume rates
- **Current Models Only**: Filters out deprecated models, defaults to gemini-2.5-pro
- **Enhanced Validation UI**: Table-based validation interface with inline editing and action buttons
- **Pairwise Comparison System**: Human and AI-powered comparison interface for Bradley-Terry analysis
- **Statistical Analysis**: Connected graph validation, temporal trend analysis, and significance testing
- **Bulk Processing**: Command-line interface for processing multiple cases
- **Data Integrity**: Robust duplicate handling and foreign key constraints

## Configuration

- Database path is hardcoded in config.py: `/Users/brandon/My Drive/Learning/Coding/SCC Research/scc_analysis_project/parquet/scc_cases.db`
- API keys are stored persistently in `.api_key.json` file (excluded from git via .gitignore)
- Default model is set to `gemini-2.5-pro` in config.py
- Updated pricing for current Gemini models (2025) including 2.5 Pro and Flash variants
- High-volume pricing is supported for Gemini 2.5 Pro (>200K tokens)

## Important Notes

- The application expects Excel metadata file at specific hardcoded path
- All AI extractions go through a validation workflow before being marked as accurate
- The system supports both individual case processing and bulk batch processing
- Bradley-Terry analysis requires pairwise comparison data (currently uses placeholder data)