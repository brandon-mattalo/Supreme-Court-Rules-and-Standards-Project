"""
Experiment Management Dashboard
Meta-level interface for creating, managing, and comparing experiments
"""

import streamlit as st
from datetime import datetime, timedelta
from config import GEMINI_MODELS, execute_sql, get_database_connection, get_gemini_model
import json
import os
import time

# Lazy imports for performance
def _get_pandas():
    import pandas as pd
    return pd

def _get_numpy():
    import numpy as np
    return np
from utils.case_management import (
    get_database_counts, load_data_from_parquet, clear_database, clear_selected_cases,
    get_case_summary, get_available_cases, filter_cases_by_criteria,
    get_experiment_selected_cases, get_available_cases_for_selection,
    add_cases_to_experiments, remove_cases_from_experiments,
    calculate_bradley_terry_comparisons, generate_bradley_terry_structure,
    get_bradley_terry_structure, get_block_summary, generate_bradley_terry_comparison_pairs
)

# Import experiment execution module
from pages import experiment_execution

def run_extraction_for_experiment(experiment_id):
    """Execute legal test extraction for an experiment"""
    try:
        # Get experiment configuration
        exp = execute_sql("SELECT * FROM v2_experiments WHERE experiment_id = ?", (experiment_id,), fetch=True)
        if not exp:
            st.error("Experiment not found")
            return
            
        exp = exp[0]
        exp_dict = dict(zip(['experiment_id', 'name', 'description', 'researcher_name', 'status', 'ai_model', 
                           'temperature', 'top_p', 'top_k', 'max_output_tokens', 
                           'extraction_strategy', 'extraction_prompt', 'comparison_prompt',
                           'system_instruction', 'cost_limit_usd', 'created_date',
                           'modified_date', 'created_by'], exp))
        
        # Get API key from session state
        if 'api_key' not in st.session_state or not st.session_state.api_key:
            st.error("API key not configured. Please set your API key in the sidebar.")
            return
            
        api_key = st.session_state.api_key
            
        # Get cases that need extraction (using global selected cases pool for now)
        cases_to_extract = execute_sql("""
            SELECT c.case_id, c.case_name, c.citation, c.case_text, c.case_length
            FROM v2_cases c
            JOIN v2_experiment_selected_cases esc ON c.case_id = esc.case_id
            WHERE c.case_id NOT IN (
                SELECT case_id FROM v2_experiment_extractions 
                WHERE experiment_id = ?
            )
        """, (experiment_id,), fetch=True)
        
        if not cases_to_extract:
            st.info("No cases need extraction for this experiment. Make sure cases are selected for experiments in the Cases section.")
            return
            
        # Load extraction prompt from experiment configuration or file
        if exp_dict.get('extraction_prompt') and exp_dict['extraction_prompt'].strip():
            extraction_prompt = exp_dict['extraction_prompt']
        else:
            # Fall back to file if experiment doesn't have custom prompt
            prompt_file = 'prompts/extractor_prompt.txt'
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    extraction_prompt = f.read()
            else:
                extraction_prompt = "Extract the main legal test from this case."
            
        # Configure Gemini model with structured output
        import google.generativeai as genai
        
        # Define the structured schema for extraction
        extraction_schema = {
            "type": "object",
            "properties": {
                "legal_test": {
                    "type": "string",
                    "description": "The legal test extracted from the case"
                },
                "passages": {
                    "type": "string", 
                    "description": "The paragraphs (e.g., paras. x, y-z) or pages (e.g., pages x, y-z) from the decision where the test is found"
                },
                "test_novelty": {
                    "type": "string",
                    "enum": ["new test", "major change in existing test", "minor change in existing test", "application of existing test", "no substantive discussion"],
                    "description": "Classification of the test novelty"
                }
            },
            "required": ["legal_test", "passages", "test_novelty"]
        }
        
        # Configure model with structured output and system instruction
        genai.configure(api_key=api_key)
        system_instruction = exp_dict.get('system_instruction', '').strip()
        if not system_instruction:
            system_instruction = "You are a helpful assistant that helps legal researchers analyze legal texts."
        
        model = genai.GenerativeModel(
            model_name=exp_dict['ai_model'],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=extraction_schema,
                temperature=exp_dict['temperature'],
                top_p=exp_dict.get('top_p', 1.0),
                top_k=exp_dict.get('top_k', 40),
                max_output_tokens=exp_dict.get('max_output_tokens', 8192)
            ),
            system_instruction=system_instruction
        )
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        total_cases = len(cases_to_extract)
        total_cost = 0.0
        
        status_placeholder.info(f"Starting extraction for {total_cases} cases...")
        
        # Process each case
        for i, case in enumerate(cases_to_extract):
            case_id, case_name, citation, case_text, case_length = case
            
            try:
                # Update progress
                progress_placeholder.progress((i + 1) / total_cases, text=f"Processing case {i + 1}/{total_cases}: {case_name}")
                
                # Prepare the prompt
                full_prompt = f"{extraction_prompt}\n\nCase Text:\n{case_text}"
                
                # Call Gemini API
                response = model.generate_content(full_prompt)
                
                # Parse structured JSON response
                try:
                    structured_response = json.loads(response.text)
                    legal_test = structured_response.get('legal_test', '')
                    passages = structured_response.get('passages', '')
                    test_novelty = structured_response.get('test_novelty', '')
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    legal_test = response.text
                    passages = "Not available"
                    test_novelty = "no substantive discussion"
                
                # Calculate cost (simplified)
                input_tokens = len(full_prompt.split()) * 1.3  # Rough token estimate
                output_tokens = len(response.text.split()) * 1.3
                model_pricing = GEMINI_MODELS.get(exp_dict['ai_model'], {'input': 0.30, 'output': 2.50})
                case_cost = (input_tokens / 1_000_000) * model_pricing['input'] + (output_tokens / 1_000_000) * model_pricing['output']
                total_cost += case_cost
                
                # Store in database with structured fields
                execute_sql("""
                    INSERT INTO v2_experiment_extractions 
                    (experiment_id, case_id, legal_test_name, legal_test_content, 
                     extraction_rationale, test_passages, test_novelty,
                     rule_like_score, confidence_score, validation_status, api_cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (experiment_id, case_id, f"Legal Test from {case_name}", legal_test,
                      "AI extracted legal test using structured output", passages, test_novelty,
                      0.5, 0.8, 'pending', case_cost))
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                st.error(f"Error processing case {case_name}: {str(e)}")
                continue
        
        # Update experiment status and cost
        execute_sql("UPDATE v2_experiments SET modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
        
        # Check if all extractions are complete
        remaining_cases = execute_sql("""
            SELECT COUNT(*) FROM v2_cases c
            JOIN v2_experiment_selected_cases esc ON c.case_id = esc.case_id
            WHERE c.case_id NOT IN (
                SELECT case_id FROM v2_experiment_extractions 
                WHERE experiment_id = ?
            )
        """, (experiment_id,), fetch=True)
        
        if remaining_cases and remaining_cases[0][0] == 0:
            # Check if comparisons are also complete
            total_comparisons = execute_sql("""
                SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = ?
            """, (experiment_id,), fetch=True)
            
            required_comparisons = calculate_bradley_terry_comparisons(len(cases_to_extract))
            
            if total_comparisons and total_comparisons[0][0] >= required_comparisons:
                execute_sql("UPDATE v2_experiments SET status = 'complete', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
        
        progress_placeholder.empty()
        status_placeholder.success(f"✅ Extraction complete! Processed {total_cases} cases. Total cost: ${total_cost:.2f}")
        
    except Exception as e:
        st.error(f"Error during extraction: {str(e)}")

def run_comparisons_for_experiment(experiment_id):
    """Execute pairwise comparisons for an experiment"""
    try:
        # Get experiment configuration
        exp = execute_sql("SELECT * FROM v2_experiments WHERE experiment_id = ?", (experiment_id,), fetch=True)
        if not exp:
            st.error("Experiment not found")
            return
            
        exp = exp[0]
        exp_dict = dict(zip(['experiment_id', 'name', 'description', 'researcher_name', 'status', 'ai_model', 
                           'temperature', 'top_p', 'top_k', 'max_output_tokens', 
                           'extraction_strategy', 'extraction_prompt', 'comparison_prompt',
                           'system_instruction', 'cost_limit_usd', 'created_date',
                           'modified_date', 'created_by'], exp))
        
        # Get API key from session state
        if 'api_key' not in st.session_state or not st.session_state.api_key:
            st.error("API key not configured. Please set your API key in the sidebar.")
            return
            
        api_key = st.session_state.api_key
            
        # Get comparison pairs that need processing
        comparison_pairs, pair_block_info = generate_bradley_terry_comparison_pairs()
        
        if not comparison_pairs:
            st.info("No comparison pairs found. Please ensure Bradley-Terry structure is generated.")
            return
            
        # Get already completed comparisons
        completed_comparisons = execute_sql("""
            SELECT extraction_id_1, extraction_id_2 FROM v2_experiment_comparisons 
            WHERE experiment_id = ?
        """, (experiment_id,), fetch=True)
        
        completed_pairs = set((comp[0], comp[1]) for comp in completed_comparisons)
        
        # Get extractions for this experiment
        extractions = execute_sql("""
            SELECT extraction_id, case_id, legal_test_content 
            FROM v2_experiment_extractions 
            WHERE experiment_id = ?
        """, (experiment_id,), fetch=True)
        
        if not extractions:
            st.error("No extractions found for this experiment. Please run extractions first.")
            return
            
        extraction_map = {ext[1]: ext for ext in extractions}  # case_id -> extraction
        
        # Load comparison prompt from experiment configuration or file
        if exp_dict.get('comparison_prompt') and exp_dict['comparison_prompt'].strip():
            comparison_prompt = exp_dict['comparison_prompt']
        else:
            # Fall back to file if experiment doesn't have custom prompt
            prompt_file = 'prompts/comparator_prompt.txt'
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    comparison_prompt = f.read()
            else:
                comparison_prompt = "Compare these two legal tests and determine which is more rule-like."
            
        # Configure Gemini model with structured output for comparisons
        import google.generativeai as genai
        
        # Define the structured schema for comparison
        comparison_schema = {
            "type": "object",
            "properties": {
                "more_rule_like_test": {
                    "type": "string",
                    "enum": ["Test A", "Test B"],
                    "description": "Which test is more rule-like"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Clear reasoning for why the chosen test is more rule-like, referring to cases as Test A and Test B only"
                }
            },
            "required": ["more_rule_like_test", "reasoning"]
        }
        
        # Configure model with structured output and system instruction
        genai.configure(api_key=api_key)
        system_instruction = exp_dict.get('system_instruction', '').strip()
        if not system_instruction:
            system_instruction = "You are a helpful assistant that helps legal researchers analyze legal texts."
        
        model = genai.GenerativeModel(
            model_name=exp_dict['ai_model'],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=comparison_schema,
                temperature=exp_dict['temperature'],
                top_p=exp_dict.get('top_p', 1.0),
                top_k=exp_dict.get('top_k', 40),
                max_output_tokens=exp_dict.get('max_output_tokens', 8192)
            ),
            system_instruction=system_instruction
        )
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        pairs_to_process = []
        for case_id_1, case_id_2 in comparison_pairs:
            if case_id_1 in extraction_map and case_id_2 in extraction_map:
                ext_1 = extraction_map[case_id_1]
                ext_2 = extraction_map[case_id_2]
                if (ext_1[0], ext_2[0]) not in completed_pairs and (ext_2[0], ext_1[0]) not in completed_pairs:
                    pairs_to_process.append((ext_1, ext_2))
        
        if not pairs_to_process:
            st.info("No comparison pairs need processing.")
            return
            
        total_pairs = len(pairs_to_process)
        total_cost = 0.0
        
        status_placeholder.info(f"Starting comparisons for {total_pairs} pairs...")
        
        # Process each pair
        for i, (ext_1, ext_2) in enumerate(pairs_to_process):
            try:
                # Update progress
                progress_placeholder.progress((i + 1) / total_pairs, text=f"Comparing pair {i + 1}/{total_pairs}")
                
                # Prepare the prompt
                full_prompt = f"{comparison_prompt}\n\nTest A: {ext_1[2]}\n\nTest B: {ext_2[2]}"
                
                # Call Gemini API
                response = model.generate_content(full_prompt)
                
                # Parse structured JSON response
                try:
                    structured_response = json.loads(response.text)
                    more_rule_like_test = structured_response.get('more_rule_like_test', 'Test A')
                    reasoning = structured_response.get('reasoning', '')
                    
                    # Determine winner based on structured response
                    winner_id = ext_1[0] if more_rule_like_test == "Test A" else ext_2[0]
                    
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    response_text = response.text
                    winner_id = ext_1[0] if "Test A" in response_text else ext_2[0]
                    reasoning = response_text
                
                # Calculate cost (simplified)
                input_tokens = len(full_prompt.split()) * 1.3
                output_tokens = len(response.text.split()) * 1.3
                model_pricing = GEMINI_MODELS.get(exp_dict['ai_model'], {'input': 0.30, 'output': 2.50})
                pair_cost = (input_tokens / 1_000_000) * model_pricing['input'] + (output_tokens / 1_000_000) * model_pricing['output']
                total_cost += pair_cost
                
                # Store in database with structured fields
                execute_sql("""
                    INSERT INTO v2_experiment_comparisons 
                    (experiment_id, extraction_id_1, extraction_id_2, winner_id, 
                     comparison_rationale, confidence_score, api_cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (experiment_id, ext_1[0], ext_2[0], winner_id, reasoning, 0.8, pair_cost))
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                st.error(f"Error processing comparison pair {i + 1}: {str(e)}")
                continue
        
        # Update experiment status
        execute_sql("UPDATE v2_experiments SET modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
        
        # Check if all comparisons are complete
        total_comparisons = execute_sql("""
            SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = ?
        """, (experiment_id,), fetch=True)
        
        required_comparisons = calculate_bradley_terry_comparisons(len(extractions))
        
        if total_comparisons and total_comparisons[0][0] >= required_comparisons:
            execute_sql("UPDATE v2_experiments SET status = 'complete', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
        
        progress_placeholder.empty()
        status_placeholder.success(f"✅ Comparisons complete! Processed {total_pairs} pairs. Total cost: ${total_cost:.2f}")
        
    except Exception as e:
        st.error(f"Error during comparisons: {str(e)}")

@st.cache_resource
def initialize_experiment_tables():
    """Initialize database tables for experiment management (cached - runs only once)"""
    
    # Cases table (v2) - Just store the cases and metadata
    from config import DB_TYPE
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_cases (
                case_id SERIAL PRIMARY KEY,
                case_name TEXT NOT NULL,
                citation TEXT UNIQUE NOT NULL,
                decision_year INTEGER,
                area_of_law TEXT,
                subject TEXT,
                decision_url TEXT,
                case_text TEXT,
                case_length INTEGER,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_cases (
                case_id INTEGER PRIMARY KEY,
                case_name TEXT NOT NULL,
                citation TEXT UNIQUE NOT NULL,
                decision_year INTEGER,
                area_of_law TEXT,
                subject TEXT,
                decision_url TEXT,
                case_text TEXT,
                case_length INTEGER,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
    
    # Experiments table (v2)
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiments (
                experiment_id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                researcher_name TEXT DEFAULT '',
                status TEXT DEFAULT 'draft',
                ai_model TEXT DEFAULT 'gemini-2.5-pro',
                temperature REAL DEFAULT 0.0,
                top_p REAL DEFAULT 1.0,
                top_k INTEGER DEFAULT 40,
                max_output_tokens INTEGER DEFAULT 8192,
                extraction_strategy TEXT DEFAULT 'single_test',
                extraction_prompt TEXT,
                comparison_prompt TEXT,
                system_instruction TEXT,
                cost_limit_usd REAL DEFAULT 100.0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT DEFAULT 'researcher'
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiments (
                experiment_id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                researcher_name TEXT DEFAULT '',
                status TEXT DEFAULT 'draft',
                ai_model TEXT DEFAULT 'gemini-2.5-pro',
                temperature REAL DEFAULT 0.0,
                top_p REAL DEFAULT 1.0,
                top_k INTEGER DEFAULT 40,
                max_output_tokens INTEGER DEFAULT 8192,
                extraction_strategy TEXT DEFAULT 'single_test',
                extraction_prompt TEXT,
                comparison_prompt TEXT,
                system_instruction TEXT,
                cost_limit_usd REAL DEFAULT 100.0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT DEFAULT 'researcher'
            );
        ''')
    
    # Experiment runs table (v2)
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_runs (
                run_id SERIAL PRIMARY KEY,
                experiment_id INTEGER,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                cases_processed INTEGER DEFAULT 0,
                tests_extracted INTEGER DEFAULT 0,
                comparisons_completed INTEGER DEFAULT 0,
                total_cost_usd REAL DEFAULT 0.0,
                execution_time_minutes REAL DEFAULT 0.0,
                error_message TEXT,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_runs (
                run_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                cases_processed INTEGER DEFAULT 0,
                tests_extracted INTEGER DEFAULT 0,
                comparisons_completed INTEGER DEFAULT 0,
                total_cost_usd REAL DEFAULT 0.0,
                execution_time_minutes REAL DEFAULT 0.0,
                error_message TEXT,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id)
            );
        ''')
    
    # Experiment extractions table (v2) - Store extractions per experiment
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_extractions (
                extraction_id SERIAL PRIMARY KEY,
                experiment_id INTEGER,
                case_id INTEGER,
                legal_test_name TEXT,
                legal_test_content TEXT,
                extraction_rationale TEXT,
                rule_like_score REAL,
                confidence_score REAL,
                validation_status TEXT DEFAULT 'pending',
                validator_notes TEXT,
                api_cost_usd REAL DEFAULT 0.0,
                extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id),
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(experiment_id, case_id)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_extractions (
                extraction_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                case_id INTEGER,
                legal_test_name TEXT,
                legal_test_content TEXT,
                extraction_rationale TEXT,
                rule_like_score REAL,
                confidence_score REAL,
                validation_status TEXT DEFAULT 'pending',
                validator_notes TEXT,
                api_cost_usd REAL DEFAULT 0.0,
                extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id),
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(experiment_id, case_id)
            );
        ''')
    
    # Experiment comparisons table (v2) - Store pairwise comparisons per experiment
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_comparisons (
                comparison_id SERIAL PRIMARY KEY,
                experiment_id INTEGER,
                extraction_id_1 INTEGER,
                extraction_id_2 INTEGER,
                winner_id INTEGER,
                comparison_rationale TEXT,
                confidence_score REAL,
                human_validated BOOLEAN DEFAULT FALSE,
                api_cost_usd REAL DEFAULT 0.0,
                comparison_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id),
                FOREIGN KEY (extraction_id_1) REFERENCES v2_experiment_extractions (extraction_id),
                FOREIGN KEY (extraction_id_2) REFERENCES v2_experiment_extractions (extraction_id),
                FOREIGN KEY (winner_id) REFERENCES v2_experiment_extractions (extraction_id),
                UNIQUE(experiment_id, extraction_id_1, extraction_id_2)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_comparisons (
                comparison_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                extraction_id_1 INTEGER,
                extraction_id_2 INTEGER,
                winner_id INTEGER,
                comparison_rationale TEXT,
                confidence_score REAL,
                human_validated BOOLEAN DEFAULT FALSE,
                api_cost_usd REAL DEFAULT 0.0,
                comparison_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id),
                FOREIGN KEY (extraction_id_1) REFERENCES v2_experiment_extractions (extraction_id),
                FOREIGN KEY (extraction_id_2) REFERENCES v2_experiment_extractions (extraction_id),
                FOREIGN KEY (winner_id) REFERENCES v2_experiment_extractions (extraction_id),
                UNIQUE(experiment_id, extraction_id_1, extraction_id_2)
            );
        ''')
    
    # Experiment results summary table (v2)
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_results (
                result_id SERIAL PRIMARY KEY,
                experiment_id INTEGER,
                metric_type TEXT,
                metric_value REAL,
                bt_statistics_json TEXT,
                regression_results_json TEXT,
                confidence_scores_json TEXT,
                calculated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_results (
                result_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                metric_type TEXT,
                metric_value REAL,
                bt_statistics_json TEXT,
                regression_results_json TEXT,
                confidence_scores_json TEXT,
                calculated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id)
            );
        ''')
    
    # Experiment selected cases table (v2) - Cases chosen for experiments
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_selected_cases (
                selection_id SERIAL PRIMARY KEY,
                case_id INTEGER,
                selected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                selected_by TEXT DEFAULT 'researcher',
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(case_id)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_selected_cases (
                selection_id INTEGER PRIMARY KEY,
                case_id INTEGER,
                selected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                selected_by TEXT DEFAULT 'researcher',
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(case_id)
            );
        ''')
    
    # Bradley-Terry structure table (v2) - Centralized block assignments for consistent methodology
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_bradley_terry_structure (
                structure_id SERIAL PRIMARY KEY,
                case_id INTEGER,
                block_number INTEGER NOT NULL,
                case_role TEXT CHECK (case_role IN ('core', 'bridge')) NOT NULL,
                importance_score REAL DEFAULT 0.0,
                assigned_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                assigned_by TEXT DEFAULT 'system',
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(case_id)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_bradley_terry_structure (
                structure_id INTEGER PRIMARY KEY,
                case_id INTEGER,
                block_number INTEGER NOT NULL,
                case_role TEXT CHECK (case_role IN ('core', 'bridge')) NOT NULL,
                importance_score REAL DEFAULT 0.0,
                assigned_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                assigned_by TEXT DEFAULT 'system',
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(case_id)
            );
        ''')
    
    # Add new columns for structured extraction data (if they don't exist)
    try:
        execute_sql('ALTER TABLE v2_experiment_extractions ADD COLUMN test_passages TEXT;')
    except:
        pass  # Column already exists
    
    try:
        execute_sql('ALTER TABLE v2_experiment_extractions ADD COLUMN test_novelty TEXT;')
    except:
        pass  # Column already exists
    
    # Add indexes for performance (v2)
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_cases_citation ON v2_cases (citation);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_cases_year ON v2_cases (decision_year);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiments_status ON v2_experiments (status);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_runs_experiment_id ON v2_experiment_runs (experiment_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_extractions_experiment_id ON v2_experiment_extractions (experiment_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_extractions_case_id ON v2_experiment_extractions (case_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_comparisons_experiment_id ON v2_experiment_comparisons (experiment_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_results_experiment_id ON v2_experiment_results (experiment_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_selected_cases_case_id ON v2_experiment_selected_cases (case_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_bradley_terry_structure_case_id ON v2_bradley_terry_structure (case_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_bradley_terry_structure_block ON v2_bradley_terry_structure (block_number);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_bradley_terry_structure_role ON v2_bradley_terry_structure (case_role);')
    
    # Additional performance indexes
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiments_modified_date ON v2_experiments (modified_date DESC);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_runs_date ON v2_experiment_runs (run_date);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_cases_case_length ON v2_cases (case_length) WHERE case_length IS NOT NULL;')

# Cache experiment overview data for 60 seconds
@st.cache_data(ttl=60)
def get_experiments_overview_data():
    """Get optimized experiment data for overview with minimal columns"""
    experiments_data = execute_sql("""
        SELECT 
            e.experiment_id, e.name, e.description, e.status, e.ai_model, 
            e.extraction_strategy, e.modified_date,
            COALESCE(
                (SELECT SUM(api_cost_usd) FROM v2_experiment_extractions WHERE experiment_id = e.experiment_id) + 
                (SELECT SUM(api_cost_usd) FROM v2_experiment_comparisons WHERE experiment_id = e.experiment_id), 
                0
            ) as total_cost,
            COALESCE((SELECT COUNT(*) FROM v2_experiment_extractions WHERE experiment_id = e.experiment_id), 0) as total_tests,
            COALESCE((SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = e.experiment_id), 0) as total_comparisons
        FROM v2_experiments e
        ORDER BY e.modified_date DESC
        LIMIT 50
    """, fetch=True)
    return experiments_data

# Cache case statistics for 120 seconds (changes less frequently)
@st.cache_data(ttl=120)
def get_case_statistics():
    """Get case count and length statistics"""
    stats = {}
    
    # Get counts
    selected_cases_count = execute_sql("SELECT COUNT(*) FROM v2_experiment_selected_cases", fetch=True)[0][0]
    total_cases_count = execute_sql("SELECT COUNT(*) FROM v2_cases", fetch=True)[0][0]
    
    # Get case length statistics with single query
    if selected_cases_count > 0:
        case_stats = execute_sql("""
            SELECT 
                AVG(CASE WHEN s.case_id IS NOT NULL THEN c.case_length END) as avg_selected_length,
                AVG(c.case_length) as avg_all_length
            FROM v2_cases c 
            LEFT JOIN v2_experiment_selected_cases s ON c.case_id = s.case_id
            WHERE c.case_length IS NOT NULL
        """, fetch=True)
        
        if case_stats and case_stats[0]:
            stats['avg_selected_case_length'] = float(case_stats[0][0]) if case_stats[0][0] else 52646.0
            stats['avg_all_case_length'] = float(case_stats[0][1]) if case_stats[0][1] else 52646.0
        else:
            stats['avg_selected_case_length'] = 52646.0
            stats['avg_all_case_length'] = 52646.0
    else:
        # If no selected cases, get overall average
        all_avg = execute_sql("SELECT AVG(case_length) FROM v2_cases WHERE case_length IS NOT NULL", fetch=True)
        avg_length = float(all_avg[0][0]) if all_avg and all_avg[0][0] else 52646.0
        stats['avg_selected_case_length'] = avg_length  
        stats['avg_all_case_length'] = avg_length
    
    stats['selected_cases_count'] = selected_cases_count
    stats['total_cases_count'] = total_cases_count
    
    return stats

# Cache CSS injection for startup performance
@st.cache_data(ttl=3600)  # 1 hour cache for CSS
def inject_sidebar_css():
    """Inject cached CSS for sidebar styling"""
    st.markdown("""
    <style>
    /* Compact sidebar styling */
    .stSidebar .stButton > button {
        padding: 0.5rem 1rem !important;
        margin: 0.2rem 0 !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
        height: auto !important;
        min-height: 2.5rem !important;
    }
    
    /* Selected navigation buttons - blue theme */
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #0066cc !important;
        color: white !important;
        border: 1px solid #004499 !important;
        box-shadow: 0 2px 4px rgba(0,102,204,0.2) !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #0056b3 !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,102,204,0.3) !important;
    }
    
    /* Action buttons - red theme */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #ff4b4b !important;
        color: white !important;
        border: 1px solid #cc0000 !important;
        box-shadow: 0 2px 4px rgba(255,75,75,0.2) !important;
    }
    div[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #e60000 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(255,75,75,0.3) !important;
    }
    
    /* Default buttons - clean styling */
    div[data-testid="stSidebar"] button:not([kind]) {
        background-color: #f8f9fa !important;
        color: #333 !important;
        border: 1px solid #dee2e6 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSidebar"] button:not([kind]):hover {
        background-color: #e9ecef !important;
        color: #000 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Experiment list styling */
    .stSidebar .stExpander {
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Compact spacing */
    .stSidebar .stMarkdown {
        margin: 0.3rem 0 !important;
    }
    
    /* Settings button special styling */
    div[data-testid="stSidebar"] button[aria-label*="Settings"] {
        padding: 0.3rem 0.5rem !important;
        font-size: 1.1rem !important;
        min-height: 2rem !important;
        border-radius: 50% !important;
        width: 2.5rem !important;
        height: 2.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache cost calculation parameters for 300 seconds (5 min - very stable)
@st.cache_data(ttl=300)
def get_cost_calculation_params(n_cases, avg_selected_case_length, extraction_strategy='single_test'):
    """Pre-calculate shared cost parameters for all experiment cards"""
    required_comparisons = calculate_bradley_terry_comparisons(n_cases)
    
    # Calculate shared cost parameters
    avg_tokens_selected = avg_selected_case_length / 4
    system_prompt_tokens = 100
    extraction_prompt_tokens = 200
    comparison_prompt_tokens = 150
    extracted_test_tokens = 325  # ~250 words * 1.3 tokens/word
    
    # Base cost per extraction for any model (includes prompt tokens)
    extraction_input_tokens = avg_tokens_selected + system_prompt_tokens + extraction_prompt_tokens
    
    # Calculate comparison input tokens based on extraction strategy
    if extraction_strategy == 'full_text_comparison':
        # For full text comparison, we compare entire case texts
        comparison_input_tokens = (avg_tokens_selected * 2) + system_prompt_tokens + comparison_prompt_tokens
    else:
        # For single_test/multi_test, we compare extracted tests
        comparison_input_tokens = (extracted_test_tokens * 2) + system_prompt_tokens + comparison_prompt_tokens
    
    comparison_output_tokens = 100  # Estimated tokens for comparison result
    
    # Bradley-Terry parameters for display
    block_size = 15  # 12 core + 3 bridge cases per block
    core_cases_per_block = 12
    comparisons_per_block = 105
    
    return {
        'required_comparisons': required_comparisons,
        'extraction_input_tokens': extraction_input_tokens,
        'extracted_test_tokens': extracted_test_tokens,
        'comparison_input_tokens': comparison_input_tokens,
        'comparison_output_tokens': comparison_output_tokens,
        'block_size': block_size,
        'core_cases_per_block': core_cases_per_block,
        'comparisons_per_block': comparisons_per_block
    }

def show_experiment_overview():
    """Display overview of all experiments"""
    st.header("📊 Experiment Overview")
    
    # Get cached experiment data
    experiments_data = get_experiments_overview_data()
    
    if not experiments_data:
        st.info("No experiments found. Create your first experiment below!")
        return
    
    # Convert to DataFrame with optimized columns
    columns = ['experiment_id', 'name', 'description', 'status', 'ai_model', 
               'extraction_strategy', 'modified_date', 'total_cost', 'total_tests', 'total_comparisons']
    
    pd = _get_pandas()
    df = pd.DataFrame(experiments_data, columns=columns)
    
    # Get cached case statistics
    try:
        stats = get_case_statistics()
        selected_cases_count = stats['selected_cases_count']
        total_cases_count = stats['total_cases_count']
        avg_selected_case_length = stats['avg_selected_case_length']
        avg_all_case_length = stats['avg_all_case_length']
        
    except Exception as e:
        selected_cases_count = 0
        total_cases_count = 0
        avg_selected_case_length = 52646.0
        avg_all_case_length = 52646.0
    
    # Calculate shared parameters once using cached function
    n_cases = selected_cases_count
    # Use default strategy for overview - individual cards will recalculate with specific strategy
    cost_params = get_cost_calculation_params(n_cases, avg_selected_case_length, 'single_test')
    
    required_comparisons = cost_params['required_comparisons']
    block_size = cost_params['block_size']
    core_cases_per_block = cost_params['core_cases_per_block']
    comparisons_per_block = cost_params['comparisons_per_block']
    
    # Skip if no cases selected
    if n_cases == 0:
        st.info("No cases selected for experiments yet. Select cases first to see experiment details.")
        return
    
    # Display experiment cards in responsive grid
    st.write(f"**{len(df)} experiments found**")
    st.write("")
    
    # Create responsive columns for card layout (3 cards per row on wide screens)
    num_cols = 3
    rows = [df.iloc[i:i+num_cols] for i in range(0, len(df), num_cols)]
    
    for row in rows:
        cols = st.columns(num_cols)
        
        for idx, (_, exp) in enumerate(row.iterrows()):
            if idx < len(cols):  # Safety check
                with cols[idx]:
                    show_experiment_card(exp, n_cases, required_comparisons, 
                                        avg_selected_case_length, avg_all_case_length, 
                                        total_cases_count, cost_params)

def show_experiment_card(exp, n_cases, required_comparisons, avg_selected_case_length, 
                        avg_all_case_length, total_cases_count, cost_params):
    """Display a single experiment card"""
            
    # Get model pricing (prices are per million tokens)
    model_pricing = GEMINI_MODELS.get(exp['ai_model'], {'input': 0.30, 'output': 2.50})
    
    # Recalculate cost parameters for this specific experiment's strategy
    exp_cost_params = get_cost_calculation_params(n_cases, avg_selected_case_length, exp['extraction_strategy'])
    
    extraction_input_tokens = exp_cost_params['extraction_input_tokens']
    extracted_test_tokens = exp_cost_params['extracted_test_tokens']
    comparison_input_tokens = exp_cost_params['comparison_input_tokens']
    comparison_output_tokens = exp_cost_params['comparison_output_tokens']
    
    # Calculate per-case costs
    extraction_cost_per_case = (extraction_input_tokens / 1_000_000) * model_pricing['input'] + (extracted_test_tokens / 1_000_000) * model_pricing['output']
    comparison_cost_per_pair = (comparison_input_tokens / 1_000_000) * model_pricing['input'] + (comparison_output_tokens / 1_000_000) * model_pricing['output']
    
    # Calculate estimates
    remaining_extractions = max(0, n_cases - int(exp['total_tests'] or 0))
    remaining_comparisons = max(0, required_comparisons - int(exp['total_comparisons'] or 0))
    
    extraction_cost_estimate = remaining_extractions * extraction_cost_per_case
    
    # Calculate total sample cost based on strategy
    if exp['extraction_strategy'] == 'full_text_comparison':
        # No extraction cost for full text comparison
        sample_total_cost = required_comparisons * comparison_cost_per_pair
    else:
        # Extraction + comparison costs
        sample_total_cost = (n_cases * extraction_cost_per_case) + (required_comparisons * comparison_cost_per_pair)
    
    # Status and progress
    status_colors = {
        'draft': '🟡 Draft',
        'active': '🟢 Active', 
        'completed': '🔵 Completed',
        'archived': '⚫ Archived'
    }
    
    # Use Streamlit's built-in container with visual separation
    with st.container(border=True):
        # Card content with smaller title
        st.markdown(f"### 🧪 #{exp['experiment_id']} {exp['name']}")
        st.markdown(f"**{status_colors.get(exp['status'], exp['status'])}**")
        
        # Key metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tests", f"{int(exp['total_tests'] or 0)}/{n_cases}")
            st.metric("Comparisons", f"{int(exp['total_comparisons'] or 0)}/{required_comparisons}")
        
        with col2:
            st.metric("Spent", f"${exp['total_cost'] or 0:.2f}")
            st.metric("Sample Est.", f"${sample_total_cost:.2f}")
        
        # Description (truncated)
        description = exp['description'] or 'No description'
        if len(description) > 60:
            description = description[:60] + "..."
        st.caption(f"**Model:** {exp['ai_model']} | **Strategy:** {exp['extraction_strategy']}")
        st.caption(f"**Description:** {description}")
        
        # Actions - only Details and Configure buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Details", key=f"details_{exp['experiment_id']}", use_container_width=True):
                st.session_state.selected_page = "Experiment Detail"
                st.session_state.selected_experiment = exp['experiment_id']
                st.rerun()
        
        with col2:
            if st.button("⚙️ Configure", key=f"config_{exp['experiment_id']}", use_container_width=True):
                st.session_state.editing_experiment = exp['experiment_id']
                st.rerun()

def show_experiment_configuration():
    """Show experiment configuration interface with two-step form"""
    st.header("⚙️ Experiment Configuration")
    
    # Initialize session state for multi-step form
    if 'config_step' not in st.session_state:
        st.session_state.config_step = 1
    
    # Check if we're editing an existing experiment
    editing_id = st.session_state.get('editing_experiment')
    
    if editing_id:
        # Load existing experiment
        exp_data = execute_sql(
            "SELECT * FROM v2_experiments WHERE experiment_id = ?", 
            (editing_id,), 
            fetch=True
        )
        if exp_data:
            exp = dict(zip(['experiment_id', 'name', 'description', 'researcher_name', 'status', 'ai_model', 
                           'temperature', 'top_p', 'top_k', 'max_output_tokens', 
                           'extraction_strategy', 'extraction_prompt', 'comparison_prompt',
                           'system_instruction', 'cost_limit_usd', 'created_date',
                           'modified_date', 'created_by'], exp_data[0]))
            
            # Pre-populate session state with existing experiment data
            if 'experiment_config' not in st.session_state:
                st.session_state.experiment_config = exp
        else:
            st.error("Experiment not found!")
            return
    else:
        # Default values for new experiment
        exp = {
            'name': '',
            'description': '',
            'researcher_name': '',
            'status': 'draft',
            'ai_model': 'gemini-2.5-pro',
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': 40,
            'max_output_tokens': 8192,
            'extraction_strategy': 'single_test',
            'extraction_prompt': '',
            'comparison_prompt': '',
            'system_instruction': 'You are a helpful assistant that helps legal researchers analyze legal texts.',
            'cost_limit_usd': 100.0
        }
        
        # Initialize session state with defaults
        if 'experiment_config' not in st.session_state:
            st.session_state.experiment_config = exp
    
    # Show appropriate step
    if st.session_state.config_step == 1:
        show_basic_info_form(exp, editing_id)
    elif st.session_state.config_step == 2:
        show_prompts_config_form(exp, editing_id)
    
def show_basic_info_form(exp, editing_id):
    """Show Step 1: Basic Information and AI Configuration"""
    # Progress indicator
    st.progress(0.5, text="Step 1 of 2: Basic Configuration")
    
    with st.form("basic_info_form"):
        # Basic Information
        st.subheader("📝 Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Experiment Name", value=exp['name'])
            researcher_name = st.text_input("Researcher's Name", value=exp['researcher_name'])
        
        with col2:
            description = st.text_area("Description", value=exp['description'])
            cost_limit = st.number_input("Cost Limit (USD)", min_value=0.0, value=float(exp['cost_limit_usd']))
        
        # AI Model Configuration
        st.subheader("🤖 AI Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            ai_model = st.selectbox("AI Model", list(GEMINI_MODELS.keys()), 
                                  index=list(GEMINI_MODELS.keys()).index(exp['ai_model']))
            temperature = st.slider(
                "Temperature", 
                0.0, 2.0, 
                float(exp['temperature']), 
                step=0.1,
                help="Controls response randomness. 0.0 = deterministic/consistent, 0.5-0.7 = balanced, 1.0+ = creative. Recommended: 0.0-0.3 for legal analysis."
            )
            top_p = st.slider(
                "Top P", 
                0.0, 1.0, 
                float(exp['top_p']), 
                step=0.1,
                help="Nucleus sampling threshold. 0.1 = very focused, 0.9 = diverse vocabulary, 1.0 = no filtering. Recommended: 0.8-1.0 for comprehensive analysis."
            )
        
        with col2:
            top_k = st.slider(
                "Top K", 
                1, 100, 
                int(exp['top_k']), 
                step=1,
                help="Number of top tokens to consider. 1 = deterministic, 20-40 = balanced, 80+ = creative. Recommended: 20-60 for legal text."
            )
            max_tokens = st.number_input(
                "Max Output Tokens", 
                min_value=1, 
                max_value=16384, 
                value=int(exp['max_output_tokens']),
                help="Maximum response length. 1000-2000 = summaries, 4000-8192 = detailed analysis, 8192 = comprehensive extraction."
            )
        
        # Extraction Strategy Selection
        st.subheader("📋 Extraction Strategy")
        extraction_strategy = st.selectbox(
            "Extraction Strategy", 
            ['single_test', 'multi_test', 'full_text_comparison'],
            index=['single_test', 'multi_test', 'full_text_comparison'].index(exp['extraction_strategy']),
            help="This determines the prompts configuration in the next step"
        )
        
        # Strategy preview/explanation
        st.info("💡 **Next Step**: You'll configure the prompts.")
        
        # Form submission
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.form_submit_button("➡️ Continue to Prompts Configuration", type="primary"):
                # Save basic info to session state
                st.session_state.experiment_config.update({
                    'name': name,
                    'description': description,
                    'researcher_name': researcher_name,
                    'cost_limit_usd': cost_limit,
                    'ai_model': ai_model,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                    'max_output_tokens': max_tokens,
                    'extraction_strategy': extraction_strategy
                })
                st.session_state.config_step = 2
                st.rerun()
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                st.session_state.editing_experiment = None
                st.session_state.config_step = 1
                st.rerun()

def show_prompts_config_form(exp, editing_id):
    """Show Step 2: Strategy-Specific Prompts Configuration"""
    # Progress indicator
    st.progress(1.0, text="Step 2 of 2: Prompts Configuration")
    
    config = st.session_state.experiment_config
    strategy = config['extraction_strategy']
    
    with st.form("prompts_config_form"):
        st.subheader(f"📝 {strategy.replace('_', ' ').title()} Configuration")
        
        # Strategy explanation
        if strategy == 'full_text_comparison':
            st.info("💡 **Full Text Comparison Strategy**: This strategy compares entire case texts directly without extraction. The extraction prompt will not be used.")
        elif strategy == 'multi_test':
            st.info("💡 **Multi-Test Strategy**: This strategy can extract multiple legal tests from a single case. The output will be structured as an array of test objects.")
        else:
            st.info("💡 **Single Test Strategy**: This strategy extracts one primary legal test per case.")
        
        # System instruction (always shown)
        system_instruction = st.text_area("System Instruction", value=config.get('system_instruction', exp['system_instruction']), height=100)
        
        # Strategy-specific prompts
        col1, col2 = st.columns(2)
        
        with col1:
            if strategy != 'full_text_comparison':
                extraction_prompt = st.text_area("Extraction Prompt", value=config.get('extraction_prompt', exp['extraction_prompt']), height=200,
                                               help="Custom prompt for legal test extraction (leave empty to use default)")
                
                # Show extraction schema
                with st.expander("📄 Extraction Structured Output Schema"):
                    st.markdown("**Gemini will be instructed to return JSON with these fields:**")
                    
                    if strategy == 'multi_test':
                        # Multi-test schema with arrays
                        st.code('''{"legal_tests": [
    {
        "legal_test": "string - The legal test extracted from the case",
        "passages": "string - The paragraphs (e.g., paras. x, y-z) or pages where the test is found",
        "test_novelty": "enum - One of: new test, major change in existing test, minor change in existing test, application of existing test, no substantive discussion"
    }
    // Additional test objects if multiple tests found
  ]}
''', language="json")
                    else:
                        # Single test schema
                        st.code('''{"legal_test": "string - The legal test extracted from the case",
 "passages": "string - The paragraphs (e.g., paras. x, y-z) or pages where the test is found",
 "test_novelty": "enum - One of: new test, major change in existing test, minor change in existing test, application of existing test, no substantive discussion"}
''', language="json")
            else:
                # Show info for full_text_comparison
                st.info("💡 Full text comparison strategy compares entire case texts directly without extraction.")
                extraction_prompt = ""  # No extraction prompt needed
        
        with col2:
            comparison_prompt = st.text_area("Comparison Prompt", value=config.get('comparison_prompt', exp['comparison_prompt']), height=200,
                                           help="Custom prompt for test comparison (leave empty to use default)")
            
            # Show comparison schema
            with st.expander("⚖️ Comparison Structured Output Schema"):
                st.markdown("**Gemini will be instructed to return JSON with these fields:**")
                st.code('''{"more_rule_like_test": "enum - Either 'Test A' or 'Test B'",
 "reasoning": "string - Clear reasoning for why the chosen test is more rule-like, referring to cases as Test A and Test B only"}
''', language="json")
        
        # Form submission
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.form_submit_button("⬅️ Back to Basic Info"):
                st.session_state.config_step = 1
                st.rerun()
        
        with col2:
            if st.form_submit_button("💾 Save Experiment", type="primary"):
                # Combine all config and save
                final_config = {
                    **config,
                    'system_instruction': system_instruction,
                    'extraction_prompt': extraction_prompt if strategy != 'full_text_comparison' else '',
                    'comparison_prompt': comparison_prompt,
                    'status': 'draft'
                }
                
                # Validate required fields
                if not final_config['name'] or not final_config['name'].strip():
                    st.error("Experiment name is required!")
                else:
                    saved_experiment_id = save_experiment(editing_id, final_config['name'], final_config['description'], 
                                     final_config['researcher_name'], final_config['status'], final_config['ai_model'], 
                                     final_config['temperature'], final_config['top_p'], final_config['top_k'], 
                                     final_config['max_output_tokens'], final_config['extraction_strategy'], 
                                     final_config['extraction_prompt'], final_config['comparison_prompt'], 
                                     final_config['system_instruction'], final_config['cost_limit_usd'])
                    
                    if saved_experiment_id:
                        st.success("Experiment saved successfully!")
                        # Clear session state and navigate to the experiment detail page
                        st.session_state.editing_experiment = None
                        st.session_state.config_step = 1
                        if 'experiment_config' in st.session_state:
                            del st.session_state.experiment_config
                        # Navigate to the experiment detail page
                        st.session_state.page_navigation = "Experiment Detail"
                        st.session_state.selected_page = "Experiment Detail"
                        st.session_state.selected_experiment = saved_experiment_id
                        st.rerun()
        
        with col3:
            if editing_id and st.form_submit_button("📋 Clone Experiment"):
                # Create a copy of the experiment
                base_name = config['name'] if config['name'] and config['name'].strip() else "Unnamed_Experiment"
                new_name = f"{base_name}_copy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                final_config = {
                    **config,
                    'system_instruction': system_instruction,
                    'extraction_prompt': extraction_prompt if strategy != 'full_text_comparison' else '',
                    'comparison_prompt': comparison_prompt,
                    'status': 'draft'
                }
                
                cloned_experiment_id = save_experiment(None, new_name, final_config['description'], 
                                 final_config['researcher_name'], final_config['status'], final_config['ai_model'], 
                                 final_config['temperature'], final_config['top_p'], final_config['top_k'], 
                                 final_config['max_output_tokens'], final_config['extraction_strategy'], 
                                 final_config['extraction_prompt'], final_config['comparison_prompt'], 
                                 final_config['system_instruction'], final_config['cost_limit_usd'])
                
                if cloned_experiment_id:
                    st.success(f"Experiment cloned as '{new_name}'!")
                    # Clear session state and navigate to the cloned experiment's detail page
                    st.session_state.editing_experiment = None
                    st.session_state.config_step = 1
                    if 'experiment_config' in st.session_state:
                        del st.session_state.experiment_config
                    # Navigate to the cloned experiment's detail page
                    st.session_state.page_navigation = "Experiment Detail"
                    st.session_state.selected_page = "Experiment Detail"
                    st.session_state.selected_experiment = cloned_experiment_id
                    st.rerun()

def save_experiment(experiment_id, name, description, researcher_name, status, ai_model, temperature, top_p, 
                   top_k, max_tokens, extraction_strategy, extraction_prompt, 
                   comparison_prompt, system_instruction, cost_limit):
    """Save experiment configuration to database"""
    try:
        if experiment_id:
            # Update existing experiment
            execute_sql("""
                UPDATE v2_experiments SET 
                    name = ?, description = ?, researcher_name = ?, ai_model = ?, temperature = ?,
                    top_p = ?, top_k = ?, max_output_tokens = ?, extraction_strategy = ?,
                    extraction_prompt = ?, comparison_prompt = ?, system_instruction = ?,
                    cost_limit_usd = ?, modified_date = CURRENT_TIMESTAMP
                WHERE experiment_id = ?
            """, (name, description, researcher_name, ai_model, temperature, top_p, top_k, max_tokens,
                  extraction_strategy, extraction_prompt, comparison_prompt, system_instruction,
                  cost_limit, experiment_id))
            return experiment_id
        else:
            # Create new experiment
            execute_sql("""
                INSERT INTO v2_experiments (name, description, researcher_name, status, ai_model, temperature, top_p,
                                       top_k, max_output_tokens, extraction_strategy, extraction_prompt,
                                       comparison_prompt, system_instruction, cost_limit_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, description, researcher_name, 'draft', ai_model, temperature, top_p, top_k, max_tokens,
                  extraction_strategy, extraction_prompt, comparison_prompt, system_instruction, cost_limit))
            
            # Get the ID of the newly created experiment
            new_experiment_id = execute_sql("""
                SELECT experiment_id FROM v2_experiments 
                WHERE name = ? AND researcher_name = ? AND created_date = (
                    SELECT MAX(created_date) FROM v2_experiments WHERE name = ? AND researcher_name = ?
                )
            """, (name, researcher_name, name, researcher_name), fetch=True)
            
            if new_experiment_id:
                experiment_id = new_experiment_id[0][0]
            else:
                return None
        
        # Clear caches after modifying experiments
        get_experiments_list.clear()
        _get_experiment_detail.clear()
        
        return experiment_id
    except Exception as e:
        st.error(f"Error saving experiment: {e}")
        return None

def show_case_management():
    """Show experiment case selection interface"""
    st.header("📚 Experiment Case Selection")
    st.markdown("*Select specific cases from the database to include in all experiments*")
    
    # Get database counts
    total_cases, selected_cases, tests_count, comparisons_count, validated_count = get_database_counts()
    
    # 1. Data Management (moved to top)
    with st.expander("1️⃣ **Data Management**", expanded=False):
        # Admin password protection
        admin_password = st.text_input("🔐 Admin Password (required for data operations)", 
                                     type="password", key="admin_password_dashboard")
        is_admin = admin_password == "scc2024admin"
        
        if not is_admin:
            st.warning("⚠️ Admin password required to access data loading and clearing functions.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Upload Data")
                uploaded_file = st.file_uploader("Choose a Parquet file", type="parquet", 
                                               disabled=not is_admin)
                if uploaded_file is not None and is_admin:
                    # Batch skip option
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        if st.button("🚀 Load Data", type="primary"):
                            with st.spinner("Loading data..."):
                                load_data_from_parquet(uploaded_file)
                                st.rerun()  # Refresh to update metrics
                    
                    with col_b:
                        start_batch = st.number_input("Start Batch", min_value=1, value=1, 
                                                    step=1, help="Skip to batch number (100 cases per batch)")
                        if st.button("⏭️ Load from Batch", type="secondary"):
                            with st.spinner(f"Loading from batch {start_batch}..."):
                                load_data_from_parquet(uploaded_file, start_batch=start_batch)
                                st.rerun()  # Refresh to update metrics
            
            with col2:
                st.subheader("🗑️ Database Management")
                
                # Clear all data button
                if st.button("🗑️ Clear All Data", disabled=not is_admin, type="secondary"):
                    if st.checkbox("I understand this will delete ALL data", key="confirm_clear_all"):
                        if clear_database():
                            st.rerun()
                
                st.divider()
                
                # Clear selected cases button
                if st.button("🎯 Clear Selected Cases Only", disabled=not is_admin, type="secondary"):
                    if st.checkbox("I understand this will clear experiment case selection", key="confirm_clear_selected"):
                        if clear_selected_cases():
                            st.rerun()
                
                st.caption("Clear Selected Cases removes experiment selection & Bradley-Terry structure while preserving main case database")
    
    # 2. Database Overview
    with st.expander("2️⃣ **Database Overview**", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cases Available", f"{total_cases:,}")
        with col2:
            st.metric("Cases Selected for Experiments", f"{selected_cases:,}")
        with col3:
            st.metric("Extracted Tests", f"{tests_count:,}")
        with col4:
            st.metric("Comparisons Made", f"{comparisons_count:,}")
        
        if selected_cases > 0:
            st.info(f"✅ **Experiment Dataset:** {selected_cases} cases selected. All experiments will run on these cases.")
        else:
            st.warning("⚠️ **No cases selected yet.** Select cases below to create your experiment dataset.")
    
    # 3. Case Selection for Experiments
    with st.expander("3️⃣ **Case Selection for Experiments**", expanded=True):
        # Show currently selected cases
        selected_cases_df = get_experiment_selected_cases()
        if not selected_cases_df.empty:
            with st.expander(f"📋 Currently Selected Cases ({len(selected_cases_df)})"):
                st.dataframe(
                    selected_cases_df[['case_name', 'citation', 'decision_year', 'area_of_law', 'selected_date']],
                    use_container_width=True
                )
        
        # Add new cases interface
        if total_cases > selected_cases:
            st.subheader("➕ Add Cases to Experiments")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Filters:**")
                
                # Get available cases for filter options
                sample_cases = get_available_cases_for_selection(limit=1000)
                
                if not sample_cases.empty:
                    # Year range filter
                    min_year = int(sample_cases['decision_year'].min())
                    max_year = int(sample_cases['decision_year'].max())
                    year_filter = st.slider("Year Range", min_year, max_year, (min_year, max_year), key="experiment_year_filter")
                    
                    # Area of law filter
                    unique_areas = sample_cases['area_of_law'].dropna().unique()
                    area_filter = st.multiselect("Areas of Law (optional)", unique_areas)
                    
                    # Number to select
                    available_count = len(get_available_cases_for_selection(
                        year_range=year_filter if year_filter != (min_year, max_year) else None,
                        areas=area_filter if area_filter else None
                    ))
                    
                    # Calculate suggested values divisible by 15
                    max_selectable = min(available_count, 100)
                    suggested_max = (max_selectable // 15) * 15
                    suggested_default = min(15, suggested_max) if suggested_max > 0 else 15
                    
                    num_to_select = st.number_input(
                        f"Number of cases to randomly select (must be divisible by 15)",
                        min_value=15, 
                        max_value=max_selectable, 
                        value=suggested_default,
                        step=15,
                        help=f"{available_count} cases available with current filters. Bradley-Terry analysis requires blocks of 15 cases (12 core + 3 bridge)."
                    )
                    
                    # Validation for multiples of 15
                    if num_to_select % 15 != 0:
                        st.error(f"⚠️ Number of cases must be divisible by 15 for Bradley-Terry block structure. Current: {num_to_select}, remainder: {num_to_select % 15}")
                        st.info(f"💡 Suggested values: {', '.join(str(i) for i in range(15, max_selectable + 1, 15) if i <= max_selectable)[:50]}...")
                        selection_valid = False
                    else:
                        blocks_needed = num_to_select // 15
                        st.success(f"✅ Valid selection: {num_to_select} cases = {blocks_needed} block{'s' if blocks_needed != 1 else ''} of 15 cases each")
                        selection_valid = True
            
            with col2:
                st.write("**Preview:**")
                
                if not sample_cases.empty:
                    # Get preview of available cases
                    preview_cases = get_available_cases_for_selection(
                        year_range=year_filter if year_filter != (min_year, max_year) else None,
                        areas=area_filter if area_filter else None,
                        limit=5
                    )
                    
                    if not preview_cases.empty:
                        st.dataframe(
                            preview_cases[['case_name', 'citation', 'decision_year', 'area_of_law']],
                            use_container_width=True
                        )
                        st.caption(f"Preview of {len(preview_cases)} cases (total available: {available_count})")
                    else:
                        st.warning("No cases available with current filters")
            
            # Add cases button (only enabled if selection is valid)
            button_disabled = not selection_valid if 'selection_valid' in locals() else num_to_select % 15 != 0
            button_label = "🎲 Randomly Select and Add Cases" if not button_disabled else "❌ Invalid Selection (Must be divisible by 15)"
            
            if st.button(button_label, type="primary", disabled=button_disabled):
                if num_to_select > 0 and num_to_select % 15 == 0:
                    with st.spinner(f"Randomly selecting {num_to_select} cases..."):
                        # Get random cases
                        new_cases = get_available_cases_for_selection(
                            year_range=year_filter if year_filter != (min_year, max_year) else None,
                            areas=area_filter if area_filter else None,
                            limit=num_to_select
                        )
                        
                        if not new_cases.empty:
                            # Add to experiments
                            success_count, duplicate_count = add_cases_to_experiments(new_cases['case_id'].tolist())
                            
                            if success_count > 0:
                                st.success(f"✅ Successfully added {success_count} cases to experiments!")
                                
                                # Check if we should generate/update Bradley-Terry structure
                                total_selected = len(get_experiment_selected_cases())
                                if total_selected % 15 == 0:
                                    with st.spinner("Generating Bradley-Terry block structure..."):
                                        success, message = generate_bradley_terry_structure(force_regenerate=True)
                                        if success:
                                            st.success(f"🎯 {message}")
                                            
                                            # Show block summary
                                            block_summary = get_block_summary()
                                            if block_summary:
                                                st.info(f"📊 Structure: {block_summary['total_blocks']} blocks, {block_summary['total_core_cases']} core cases, {block_summary['total_bridge_cases']} bridge cases")
                                        else:
                                            st.warning(f"⚠️ Bradley-Terry structure generation failed: {message}")
                                
                                # Show added cases
                                with st.expander("📋 Cases Added"):
                                    st.dataframe(
                                        new_cases[['case_name', 'citation', 'decision_year', 'area_of_law']],
                                        use_container_width=True
                                    )
                                
                                st.rerun()
                            else:
                                st.error("Failed to add cases to experiments")
                        else:
                            st.error("No cases found matching the criteria")
                else:
                    st.error("Please select at least 1 case")
        else:
            st.info("All available cases have been selected for experiments")

def show_experiment_comparison():
    """Show cross-experiment comparison interface"""
    st.header("📈 Experiment Comparison")
    st.markdown("*Compare methodology effectiveness across different experimental configurations*")
    
    # Get experiments with results
    experiments_with_results = execute_sql("""
        SELECT 
            e.experiment_id, 
            e.name, 
            e.ai_model, 
            e.temperature, 
            e.extraction_strategy,
            e.modified_date,
            COUNT(er.run_id) as run_count,
            SUM(er.tests_extracted) as total_tests,
            SUM(er.comparisons_completed) as total_comparisons,
            AVG(er.total_cost_usd) as avg_cost,
            MAX(er.run_date) as last_run
        FROM v2_experiments e
        LEFT JOIN v2_experiment_runs er ON e.experiment_id = er.experiment_id
        WHERE e.status IN ('active', 'completed') 
        GROUP BY e.experiment_id, e.name, e.ai_model, e.temperature, e.extraction_strategy, e.modified_date
        HAVING COUNT(er.run_id) > 0
        ORDER BY e.modified_date DESC
    """, fetch=True)
    
    if not experiments_with_results:
        st.info("No experiments with results available for comparison. Run some experiments first!")
        return
    
    # Convert to DataFrame for easier handling
    columns = ['experiment_id', 'name', 'ai_model', 'temperature', 'extraction_strategy', 
               'modified_date', 'run_count', 'total_tests', 'total_comparisons', 'avg_cost', 'last_run']
    pd = _get_pandas()
    exp_df = pd.DataFrame(experiments_with_results, columns=columns)
    
    # Experiment Selection
    st.subheader("🎯 Select Experiments to Compare")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Multi-select with experiment details
        exp_options = {}
        for _, exp in exp_df.iterrows():
            label = f"{exp['name']} ({exp['ai_model']}, temp: {exp['temperature']}, {exp['extraction_strategy']})"
            exp_options[label] = exp['experiment_id']
        
        selected_experiments = st.multiselect(
            "Choose experiments to compare:",
            exp_options.keys(),
            help="Select 2 or more experiments to compare their performance"
        )
    
    with col2:
        if len(selected_experiments) >= 2:
            st.success(f"✅ {len(selected_experiments)} experiments selected")
            show_comparison = st.button("📊 Generate Comparison", type="primary")
        else:
            st.warning("Select at least 2 experiments")
            show_comparison = False
    
    if show_comparison and len(selected_experiments) >= 2:
        # Get selected experiment IDs
        selected_ids = [exp_options[exp] for exp in selected_experiments]
        
        # Performance Comparison
        st.subheader("⚡ Performance Comparison")
        
        # Create comparison metrics
        comparison_data = []
        for exp_id in selected_ids:
            exp_info = exp_df[exp_df['experiment_id'] == exp_id].iloc[0]
            
            # Get detailed statistics (placeholder for now)
            comparison_data.append({
                'Experiment': exp_info['name'],
                'Model': exp_info['ai_model'],
                'Temperature': exp_info['temperature'],
                'Strategy': exp_info['extraction_strategy'],
                'Total Tests': exp_info['total_tests'],
                'Total Comparisons': exp_info['total_comparisons'],
                'Avg Cost ($)': f"${exp_info['avg_cost']:.2f}" if exp_info['avg_cost'] else "N/A",
                'Run Count': exp_info['run_count'],
                'Tests per Run': exp_info['total_tests'] / exp_info['run_count'] if exp_info['run_count'] > 0 else 0
            })
        
        pd = _get_pandas()
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparisons
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Tests Extracted")
            chart_data = comparison_df.set_index('Experiment')['Total Tests']
            st.bar_chart(chart_data)
        
        with col2:
            st.subheader("💰 Cost Comparison")
            # Convert cost back to numeric for charting
            cost_data = comparison_df.copy()
            cost_data['Cost_Numeric'] = cost_data['Avg Cost ($)'].str.replace('$', '').str.replace('N/A', '0').astype(float)
            chart_data = cost_data.set_index('Experiment')['Cost_Numeric']
            st.bar_chart(chart_data)
        
        # Model Performance Analysis
        st.subheader("🧠 Model Performance Analysis")
        
        # Group by model type
        model_performance = comparison_df.groupby('Model').agg({
            'Total Tests': 'sum',
            'Total Comparisons': 'sum',
            'Tests per Run': 'mean'
        }).round(2)
        
        st.write("**Performance by AI Model:**")
        st.dataframe(model_performance, use_container_width=True)
        
        # Temperature Analysis
        st.subheader("🌡️ Temperature Impact Analysis")
        
        temp_analysis = comparison_df.groupby('Temperature').agg({
            'Total Tests': 'mean',
            'Tests per Run': 'mean'
        }).round(2)
        
        st.write("**Performance by Temperature Setting:**")
        st.dataframe(temp_analysis, use_container_width=True)
        
        # Statistical Significance Testing (Placeholder)
        st.subheader("📈 Statistical Analysis")
        
        st.info("**Coming Soon:** Statistical significance testing between experiments")
        st.write("Future features will include:")
        st.write("- Bradley-Terry score comparisons")
        st.write("- Confidence intervals for performance metrics")
        st.write("- ANOVA testing for model differences")
        st.write("- Cost-effectiveness ratios")
        st.write("- Reproducibility metrics")
        
        # Export Comparison Results
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export comparison table
            csv_data = comparison_df.to_csv(index=False)
            st.download_button(
                label="📊 Export Comparison Table",
                data=csv_data,
                file_name=f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export model performance summary
            model_csv = model_performance.to_csv()
            st.download_button(
                label="🧠 Export Model Analysis",
                data=model_csv,
                file_name=f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Experiment History and Trends
    st.subheader("📅 Experiment History")
    
    if not exp_df.empty:
        # Timeline visualization
        st.write("**Recent Experiment Activity:**")
        
        # Convert last_run to datetime for plotting
        pd = _get_pandas()
        exp_df['last_run_date'] = pd.to_datetime(exp_df['last_run'])
        recent_activity = exp_df.sort_values('last_run_date', ascending=False).head(10)
        
        # Display recent activity
        for _, exp in recent_activity.iterrows():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{exp['name']}**")
                st.caption(f"{exp['ai_model']} | {exp['extraction_strategy']}")
            
            with col2:
                st.metric("Tests", int(exp['total_tests']))
            
            with col3:
                st.metric("Runs", int(exp['run_count']))
            
            with col4:
                last_run_str = exp['last_run'][:10] if exp['last_run'] else "N/A"
                st.write(f"Last: {last_run_str}")
            
            st.divider()
    
    else:
        st.info("No experiment history available.")

# Cache experiments list for 5 minutes (longer cache for startup performance)
@st.cache_data(ttl=300)
def get_experiments_list():
    """Get list of experiments for navigation (reuses overview data for efficiency)"""
    try:
        # Reuse the cached overview data to avoid duplicate queries
        experiments_data = get_experiments_overview_data()
        
        if experiments_data:
            # Extract only the fields needed for navigation
            return [{'id': exp[0], 'name': exp[1], 'status': exp[3]} for exp in experiments_data]
        return []
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
        return []

def show_sidebar_navigation():
    """Display hierarchical sidebar navigation"""
    # Initialize navigation state
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Library Overview"  # Default to Library Overview
    if 'selected_experiment' not in st.session_state:
        st.session_state.selected_experiment = None
    
    # Inject cached CSS for compact application-like sidebar
    inject_sidebar_css()
    
    # Navigation section header
    st.sidebar.markdown("**📋 Navigation**")
    
    # 1. Create New Experiment (At the top)
    if st.sidebar.button("➕ Create New Experiment", use_container_width=True, type="primary"):
        st.session_state.selected_page = "Create Experiment"
        st.session_state.selected_experiment = None
        st.session_state.editing_experiment = None  # Clear edit state
        st.rerun()
    
    # Small separator
    st.sidebar.markdown("")
    
    # 2. Experiment Library (Expandable)
    with st.sidebar.expander("🧪 Experiment Library", expanded=True):
        # Overview option
        is_active = st.session_state.selected_page == "Library Overview"
        button_kwargs = {"use_container_width": True}
        if is_active:
            button_kwargs["type"] = "secondary"
        
        if st.button("📊 Library Overview", **button_kwargs):
            st.session_state.selected_page = "Library Overview"
            st.session_state.selected_experiment = None
            st.rerun()
            st.session_state.selected_page = "Library Overview"
            st.session_state.selected_experiment = None
            st.rerun()
        
        # Individual experiments - show by default
        experiments = get_experiments_list()
        if experiments:
            st.markdown("*Experiments:*")
            for exp in experiments:
                status_emoji = {
                    'draft': '🟡',
                    'in_progress': '🟠', 
                    'complete': '🟢',
                    'archived': '⚫'
                }.get(exp['status'], '⚪')
                
                # Add experiment number and truncate long names for better UI
                display_name = exp['name'][:20] + "..." if len(exp['name']) > 20 else exp['name']
                button_label = f"{status_emoji} #{exp['id']} {display_name}"
                is_selected = (st.session_state.selected_page == "Experiment Detail" and 
                             st.session_state.selected_experiment == exp['id'])
                
                button_kwargs = {"use_container_width": True, "key": f"exp_{exp['id']}"}
                if is_selected:
                    button_kwargs["type"] = "secondary"
                
                if st.button(button_label, **button_kwargs):
                    st.session_state.selected_page = "Experiment Detail"
                    st.session_state.selected_experiment = exp['id']
                    st.rerun()
        else:
            st.caption("No experiments found")
    
    # Small separator
    st.sidebar.markdown("")
    
    # 3. Experiment Comparison
    is_active = st.session_state.selected_page == "Comparison"
    button_kwargs = {"use_container_width": True}
    if is_active:
        button_kwargs["type"] = "secondary"
    
    if st.sidebar.button("📈 Experiment Comparison", **button_kwargs):
        st.session_state.selected_page = "Comparison"
        st.session_state.selected_experiment = None
        st.rerun()
    
    # Small separator
    st.sidebar.markdown("")
    
    # 4. Cases (Blue button)
    is_active = st.session_state.selected_page == "Cases"
    button_kwargs = {"use_container_width": True}
    if is_active:
        button_kwargs["type"] = "secondary"
    
    if st.sidebar.button("📚 View/Add Cases to Experiments", **button_kwargs):
        st.session_state.selected_page = "Cases"
        st.session_state.selected_experiment = None
        st.rerun()
    
    # Small separator
    st.sidebar.markdown("")
    
    # 5. API Key Management (Collapsible)
    # Check if API key is required (from execution buttons)
    api_key_required = st.session_state.get('api_key_required', False)
    
    # Force expand if API key is required
    with st.sidebar.expander("🔑 API Key", expanded=api_key_required):
        # Check current API key status
        api_key_status = "Not set"
        if 'api_key' in st.session_state and st.session_state.api_key:
            api_key_status = f"Set (ends with ...{st.session_state.api_key[-4:]})"
        
        # Show highlighted status if API key is required
        if api_key_required:
            # Add pulsing animation CSS
            st.markdown("""
            <style>
            @keyframes pulse {
                0% { border-color: #ffeaa7; }
                50% { border-color: #ff6b6b; }
                100% { border-color: #ffeaa7; }
            }
            .api-key-required {
                animation: pulse 2s infinite;
                background-color: #fff3cd;
                padding: 10px;
                border-radius: 5px;
                border: 2px solid #ffeaa7;
            }
            </style>
            <div class="api-key-required">
                <strong>⚠️ API Key Required!</strong><br>
                Please enter your API key to execute experiments.
            </div>
            """, unsafe_allow_html=True)
            st.write("")
        
        st.write(f"**Status:** {api_key_status}")
        
        # API Key input
        api_key_input = st.text_input("Enter API Key:", type="password", key="api_key_input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save", use_container_width=True, key="save_api_key"):
                if api_key_input.strip():
                    st.session_state.api_key = api_key_input.strip()
                    # Clear the required flag once saved
                    if 'api_key_required' in st.session_state:
                        del st.session_state.api_key_required
                    st.success("API key saved!")
                    st.rerun()
                else:
                    st.error("Please enter an API key")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_api_key"):
                if 'api_key' in st.session_state:
                    del st.session_state.api_key
                st.success("API key cleared!")
                st.rerun()
        
        st.caption("API key is stored only for this session")

# Cache experiment details for 60 seconds
@st.cache_data(ttl=60)
def _get_experiment_detail(experiment_id):
    """Get experiment details from database"""
    exp_data = execute_sql(
        "SELECT experiment_id, name, description, researcher_name, status, ai_model, temperature, top_p, top_k, max_output_tokens, extraction_strategy, extraction_prompt, comparison_prompt, system_instruction, cost_limit_usd, created_date, modified_date, created_by FROM v2_experiments WHERE experiment_id = ?", 
        (experiment_id,), 
        fetch=True
    )
    
    if not exp_data:
        return None
    
    # Convert row to dictionary, handling any data type conversion needed
    row = exp_data[0]
    return {
        'experiment_id': row[0],
        'name': row[1], 
        'description': row[2],
        'researcher_name': row[3],
        'status': row[4],
        'ai_model': row[5],
        'temperature': float(row[6]) if row[6] is not None else 0.0,
        'top_p': float(row[7]) if row[7] is not None else 1.0,
        'top_k': int(row[8]) if row[8] is not None else 40,
        'max_output_tokens': int(row[9]) if row[9] is not None else 8192,
        'extraction_strategy': row[10],
        'extraction_prompt': row[11],
        'comparison_prompt': row[12], 
        'system_instruction': row[13],
        'cost_limit_usd': float(row[14]) if row[14] is not None else 100.0,
        'created_date': row[15],
        'modified_date': row[16],
        'created_by': row[17]
    }

def show_experiment_detail(experiment_id):
    """Show details for a specific experiment with comprehensive cost analysis"""
    try:
        exp = _get_experiment_detail(experiment_id)
        
        if not exp:
            st.error("Experiment not found!")
            return
        
        st.header(f"Experiment #{exp['experiment_id']}: {exp['name']}")
        
        # Status with color
        status_colors = {
            'draft': '🟡 Draft',
            'in_progress': '🟠 In Progress',
            'complete': '🟢 Complete',
            'archived': '⚫ Archived'
        }
        st.markdown(f"**Status:** {status_colors.get(exp['status'], exp['status'])}")
        
        # Get shared data for cost calculations using cached functions
        try:
            stats = get_case_statistics()
            selected_cases_count = stats['selected_cases_count']
            total_cases_count = stats['total_cases_count']
            avg_selected_case_length = stats['avg_selected_case_length']
            avg_all_case_length = stats['avg_all_case_length']
        except Exception as e:
            st.warning(f"Could not load case statistics: {e}")
            selected_cases_count = 0
            total_cases_count = 0
            avg_selected_case_length = 52646.0
            avg_all_case_length = 52646.0
        
        # Calculate comprehensive cost data
        n_cases = selected_cases_count
        required_comparisons = calculate_bradley_terry_comparisons(n_cases)
        
        # Bradley-Terry parameters
        block_size = 15
        core_cases_per_block = 12
        comparisons_per_block = 105
        
        # Get run statistics with safe handling
        try:
            runs_data = execute_sql("""
                SELECT 
                    1 as run_count,
                    (SELECT COUNT(*) FROM v2_experiment_extractions WHERE experiment_id = ?) as total_tests,
                    (SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = ?) as total_comparisons,
                    (SELECT COALESCE(SUM(api_cost_usd), 0) FROM v2_experiment_extractions WHERE experiment_id = ?) + 
                    (SELECT COALESCE(SUM(api_cost_usd), 0) FROM v2_experiment_comparisons WHERE experiment_id = ?) as total_cost
            """, (experiment_id, experiment_id, experiment_id, experiment_id), fetch=True)
            
            if runs_data and runs_data[0]:
                row = runs_data[0]
                # Handle both tuple and row object access
                if hasattr(row, '__getitem__'):
                    total_tests = int(row[1]) if row[1] is not None else 0
                    total_comparisons = int(row[2]) if row[2] is not None else 0
                    total_cost = float(row[3]) if row[3] is not None else 0.0
                else:
                    total_tests = 0
                    total_comparisons = 0 
                    total_cost = 0.0
            else:
                total_tests = 0
                total_comparisons = 0
                total_cost = 0.0
        except Exception as e:
            st.warning(f"Could not load run statistics: {e}")
            total_tests = 0
            total_comparisons = 0
            total_cost = 0.0
        
        # Layout in tabs for organization
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📋 Configuration & Stats", "💰 Cost Estimates", "🚀 Execution", "📄 Extractions", "⚖️ Comparisons", "📊 Results", "⚙️ Settings"])
        
        with tab1:
            # Configuration and Stats combined
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Information")
                st.write(f"**Description:** {exp['description'] or 'No description'}")
                st.write(f"**Researcher:** {exp.get('researcher_name', 'Not specified')}")
                # Handle datetime objects properly
                created_date = exp['created_date']
                modified_date = exp['modified_date']
                
                if isinstance(created_date, str):
                    created_str = created_date[:10] if created_date else 'Unknown'
                else:
                    created_str = created_date.strftime('%Y-%m-%d') if created_date else 'Unknown'
                
                if isinstance(modified_date, str):
                    modified_str = modified_date[:10] if modified_date else 'Unknown'
                else:
                    modified_str = modified_date.strftime('%Y-%m-%d') if modified_date else 'Unknown'
                
                st.write(f"**Created:** {created_str}")
                st.write(f"**Modified:** {modified_str}")
                
                st.subheader("Current Progress")
                st.metric("Tests Extracted", f"{total_tests}/{n_cases}")
                st.metric("Comparisons Made", f"{total_comparisons}/{required_comparisons}")
                st.metric("Total Spent", f"${total_cost:.2f}")
                
                if n_cases > 0:
                    extraction_progress = total_tests / n_cases
                    st.progress(extraction_progress, text=f"Extraction Progress: {extraction_progress:.1%}")
                
                if required_comparisons > 0:
                    comparison_progress = total_comparisons / required_comparisons
                    st.progress(comparison_progress, text=f"Comparison Progress: {comparison_progress:.1%}")
            
            with col2:
                st.subheader("AI Configuration")
                st.write(f"**Model:** {exp['ai_model']}")
                st.write(f"**Temperature:** {exp['temperature']}")
                st.write(f"**Top P:** {exp.get('top_p', 1.0)}")
                st.write(f"**Top K:** {exp.get('top_k', 40)}")
                st.write(f"**Max Tokens:** {exp.get('max_output_tokens', 8192)}")
                st.write(f"**Strategy:** {exp['extraction_strategy']}")
                st.write(f"**Cost Limit:** ${exp['cost_limit_usd']}")
                
                # Show comparison strategy
                if n_cases <= block_size:
                    comparison_strategy = "Full pairwise"
                else:
                    blocks_needed = (n_cases + core_cases_per_block - 1) // core_cases_per_block
                    comparison_strategy = f"Bradley-Terry ({blocks_needed} blocks)"
                
                st.subheader("Comparison Strategy")
                st.write(f"**Method:** {comparison_strategy}")
                st.write(f"**Required Comparisons:** {required_comparisons:,}")
            
            # Add full prompts display
            st.subheader("🤖 AI Prompts & Configuration")
            
            # System instruction
            with st.expander("📋 System Instruction"):
                system_instruction = exp.get('system_instruction', '').strip()
                if not system_instruction:
                    system_instruction = "You are a helpful assistant that helps legal researchers analyze legal texts."
                st.code(system_instruction, language="text")
            
            # Extraction prompt
            with st.expander("📄 Full Extraction Prompt"):
                extraction_prompt = exp.get('extraction_prompt', '').strip()
                if not extraction_prompt:
                    # Load default from file
                    prompt_file = 'prompts/extractor_prompt.txt'
                    if os.path.exists(prompt_file):
                        with open(prompt_file, 'r') as f:
                            extraction_prompt = f.read()
                    else:
                        extraction_prompt = "Extract the main legal test from this case."
                
                st.markdown("**Custom Prompt:**")
                st.code(extraction_prompt, language="text")
                
                st.markdown("**Structured Output Schema:**")
                st.code('''{"legal_test": "string - The legal test extracted from the case",
 "passages": "string - The paragraphs (e.g., paras. x, y-z) or pages where the test is found",
 "test_novelty": "enum - One of: new test, major change in existing test, minor change in existing test, application of existing test, no substantive discussion"}
''', language="json")
            
            # Comparison prompt
            with st.expander("⚖️ Full Comparison Prompt"):
                comparison_prompt = exp.get('comparison_prompt', '').strip()
                if not comparison_prompt:
                    # Load default from file
                    prompt_file = 'prompts/comparator_prompt.txt'
                    if os.path.exists(prompt_file):
                        with open(prompt_file, 'r') as f:
                            comparison_prompt = f.read()
                    else:
                        comparison_prompt = "Compare these two legal tests and determine which is more rule-like."
                
                st.markdown("**Custom Prompt:**")
                st.code(comparison_prompt, language="text")
                
                st.markdown("**Structured Output Schema:**")
                st.code('''{"more_rule_like_test": "enum - Either 'Test A' or 'Test B'",
 "reasoning": "string - Clear reasoning for why the chosen test is more rule-like, referring to cases as Test A and Test B only"}
''', language="json")
        
        with tab2:
            st.subheader("💰 Comprehensive Cost Analysis")
            
            # Get model pricing
            model_pricing = GEMINI_MODELS.get(exp['ai_model'], {'input': 0.30, 'output': 2.50})
            
            # Calculate detailed cost parameters for this experiment's strategy
            cost_params = get_cost_calculation_params(n_cases, avg_selected_case_length, exp['extraction_strategy'])
            extraction_input_tokens = cost_params['extraction_input_tokens']
            extracted_test_tokens = cost_params['extracted_test_tokens']
            comparison_input_tokens = cost_params['comparison_input_tokens']
            comparison_output_tokens = cost_params['comparison_output_tokens']
            
            # Per-case extraction cost
            extraction_cost_per_case = (extraction_input_tokens / 1_000_000) * model_pricing['input'] + (extracted_test_tokens / 1_000_000) * model_pricing['output']
            
            # Per-pair comparison cost based on strategy
            comparison_cost_per_pair = (comparison_input_tokens / 1_000_000) * model_pricing['input'] + (comparison_output_tokens / 1_000_000) * model_pricing['output']
            
            # Pre-calculate all values based on strategy
            if exp['extraction_strategy'] == 'full_text_comparison':
                # No extraction cost for full text comparison
                sample_extraction_cost = 0
                sample_comparison_cost = required_comparisons * comparison_cost_per_pair
                sample_total_cost = sample_comparison_cost
            else:
                # Extraction + comparison costs
                sample_extraction_cost = n_cases * extraction_cost_per_case
                sample_comparison_cost = required_comparisons * comparison_cost_per_pair
                sample_total_cost = sample_extraction_cost + sample_comparison_cost
            
            remaining_extractions = max(0, n_cases - total_tests)
            remaining_comparisons = max(0, required_comparisons - total_comparisons)
            remaining_extraction_cost = remaining_extractions * extraction_cost_per_case
            remaining_comparison_cost = remaining_comparisons * comparison_cost_per_pair
            remaining_total_cost = remaining_extraction_cost + remaining_comparison_cost
            
            # For full DB estimates, use same strategy as experiment for fair comparison
            full_db_comparisons = calculate_bradley_terry_comparisons(total_cases_count)
            full_db_cost_params = get_cost_calculation_params(total_cases_count, avg_selected_case_length, exp['extraction_strategy'])
            full_db_comparison_cost_per_pair = (full_db_cost_params['comparison_input_tokens'] / 1_000_000) * model_pricing['input'] + (full_db_cost_params['comparison_output_tokens'] / 1_000_000) * model_pricing['output']
            
            if exp['extraction_strategy'] == 'full_text_comparison':
                # No extraction cost for full text comparison
                full_extraction_cost = 0
                full_comparison_cost = full_db_comparisons * full_db_comparison_cost_per_pair
                full_total_cost = full_comparison_cost
            else:
                # Extraction + comparison costs
                full_extraction_cost = total_cases_count * extraction_cost_per_case
                full_comparison_cost = full_db_comparisons * full_db_comparison_cost_per_pair
                full_total_cost = full_extraction_cost + full_comparison_cost
            efficiency_ratio = (sample_total_cost / full_total_cost) * 100 if full_total_cost > 0 else 0
            
            # 2x2 Grid Layout for perfect alignment
            # Top row
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Sample Cost Breakdown")
                st.write(f"**Selected Cases:** {n_cases:,}")
                st.write(f"**Required Comparisons:** {required_comparisons:,}")
                st.metric("Sample Extraction Cost", f"${sample_extraction_cost:.2f}")
                st.metric("Sample Comparison Cost", f"${sample_comparison_cost:.2f}")
                st.metric("Sample Total Cost", f"${sample_total_cost:.2f}")
            
            with col2:
                st.subheader("🌍 Full Database Estimates")
                st.write(f"**Total Cases in Database:** {total_cases_count:,}")
                st.write(f"**Required Comparisons:** {full_db_comparisons:,}")
                st.metric("Full DB Extraction Cost", f"${full_extraction_cost:.2f}")
                st.metric("Full DB Comparison Cost", f"${full_comparison_cost:.2f}")
                st.metric("Full DB Total Cost", f"${full_total_cost:.2f}")
            
            # Divider
            st.write("---")
            
            # Bottom row - aligned sections
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("🎯 Remaining Work")
                st.metric("Remaining Extraction Cost", f"${remaining_extraction_cost:.2f}")
                st.metric("Remaining Comparison Cost", f"${remaining_comparison_cost:.2f}")
                st.metric("Remaining Total Cost", f"${remaining_total_cost:.2f}")
            
            with col4:
                st.subheader("📊 Cost Efficiency")
                st.metric("Sample vs Full DB", f"{efficiency_ratio:.1f}%")
                
                # Bradley-Terry efficiency
                if total_cases_count > 15:
                    full_pairwise = (total_cases_count * (total_cases_count - 1)) // 2
                    bt_efficiency = ((full_pairwise - full_db_comparisons) / full_pairwise) * 100
                    st.metric("Bradley-Terry Savings", f"{bt_efficiency:.1f}%")
                else:
                    st.metric("Bradley-Terry Savings", "N/A")
            
            # Detailed cost calculation breakdown
            st.write("---")
            with st.expander("🔍 Detailed Cost Calculation"):
                st.write("**Token Calculations:**")
                st.write(f"- Average case length: {avg_selected_case_length:,.0f} characters")
                st.write(f"- Estimated tokens per case: {avg_selected_case_length / 4:,.0f} tokens")
                st.write(f"- System prompt tokens: {100}")
                st.write(f"- Extraction prompt tokens: {200}")
                st.write(f"- Comparison prompt tokens: {150}")
                st.write(f"- Total input tokens per case: {extraction_input_tokens:,.0f}")
                st.write(f"- Expected output tokens per case: {extracted_test_tokens}")
                
                st.write("**Model Pricing:**")
                st.write(f"- Model: {exp['ai_model']}")
                st.write(f"- Input cost: ${model_pricing['input']:.2f} per million tokens")
                st.write(f"- Output cost: ${model_pricing['output']:.2f} per million tokens")
                
                if exp['extraction_strategy'] == 'full_text_comparison':
                    st.write("**Per-Case Extraction Cost:**")
                    st.write("- **No extraction needed** (comparing full case texts directly)")
                    st.write("- **Total per case: $0.00**")
                else:
                    st.write("**Per-Case Extraction Cost:**")
                    input_cost = (extraction_input_tokens / 1_000_000) * model_pricing['input']
                    output_cost = (extracted_test_tokens / 1_000_000) * model_pricing['output']
                    st.write(f"- Input cost: ${input_cost:.4f}")
                    st.write(f"- Output cost: ${output_cost:.4f}")
                    st.write(f"- **Total per case: ${extraction_cost_per_case:.4f}**")
                
                st.write("**Per-Pair Comparison Cost:**")
                if exp['extraction_strategy'] == 'full_text_comparison':
                    st.write(f"- Strategy: Full text comparison (comparing entire case texts)")
                    st.write(f"- Input tokens per comparison: {comparison_input_tokens:,.0f} (2 × {avg_selected_case_length / 4:,.0f} + prompts)")
                else:
                    st.write(f"- Strategy: {exp['extraction_strategy']} (comparing extracted tests)")
                    st.write(f"- Input tokens per comparison: {comparison_input_tokens:,.0f} (2 × {extracted_test_tokens} + prompts)")
                comparison_input_cost = (comparison_input_tokens / 1_000_000) * model_pricing['input']
                comparison_output_cost = (comparison_output_tokens / 1_000_000) * model_pricing['output']
                st.write(f"- Input cost: ${comparison_input_cost:.4f}")
                st.write(f"- Output cost: ${comparison_output_cost:.4f}")
                st.write(f"- **Total per pair: ${comparison_cost_per_pair:.4f}**")
        
        with tab3:
            # Execution interface
            st.subheader("🚀 Run Experiment")
            
            # Status check
            if exp['status'] == 'archived':
                st.error("❌ This experiment is archived and cannot be executed.")
                st.stop()
            
            # Check if API key is required and show prominent notification
            if st.session_state.get('api_key_required', False):
                st.error("⚠️ **API Key Required!** Please set your API key in the sidebar to execute experiments.")
                st.markdown("""
                <div style="background-color: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb; margin: 10px 0;">
                    <strong>🔑 How to set your API key:</strong><br>
                    1. Look at the sidebar on the left<br>
                    2. Click on "🔑 API Key" to expand it<br>
                    3. Enter your Gemini API key<br>
                    4. Click "💾 Save"
                </div>
                """, unsafe_allow_html=True)
            
            # Execution overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Extraction Phase**")
                if total_tests == 0:
                    st.write("🟡 Ready to extract legal tests from cases")
                    st.write(f"(0/{n_cases} complete)")
                elif total_tests < n_cases:
                    st.write(f"🟠 In progress: {total_tests}/{n_cases} cases processed")
                    st.write(f"({total_tests}/{n_cases} complete)")
                else:
                    st.write("🟢 All cases extracted")
                    st.write(f"({total_tests}/{n_cases} complete)")
                
                if st.button("▶️ Run Extraction", 
                           type="primary" if total_tests < n_cases else "secondary",
                           use_container_width=True,
                           disabled=(exp['status'] == 'archived')):
                    # Check if API key is set
                    if 'api_key' not in st.session_state or not st.session_state.api_key:
                        # Force expand sidebar and highlight API key section
                        st.session_state.api_key_required = True
                        st.error("⚠️ API Key must be entered to execute extractions. Please set your API key in the sidebar.")
                        st.rerun()
                    else:
                        # Update experiment status to in_progress
                        execute_sql("UPDATE v2_experiments SET status = 'in_progress', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
                        
                        # Execute extraction
                        run_extraction_for_experiment(experiment_id)
                        st.rerun()
                    
            with col2:
                st.write("**Comparison Phase**")
                if total_comparisons == 0:
                    if total_tests == n_cases:
                        st.write("🟡 Ready to run pairwise comparisons")
                        st.write(f"(0/{required_comparisons} complete)")
                    else:
                        st.write("⏳ Waiting for extractions to complete")
                        st.write(f"(0/{required_comparisons} complete)")
                elif total_comparisons < required_comparisons:
                    st.write(f"🟠 In progress: {total_comparisons}/{required_comparisons} comparisons")
                    st.write(f"({total_comparisons}/{required_comparisons} complete)")
                else:
                    st.write("🟢 All comparisons completed")
                    st.write(f"({total_comparisons}/{required_comparisons} complete)")
                
                comparison_disabled = (exp['status'] == 'archived') or (total_tests < n_cases)
                if st.button("▶️ Run Comparisons", 
                           type="primary" if total_comparisons < required_comparisons and not comparison_disabled else "secondary",
                           use_container_width=True,
                           disabled=comparison_disabled):
                    # Check if API key is set
                    if 'api_key' not in st.session_state or not st.session_state.api_key:
                        # Force expand sidebar and highlight API key section
                        st.session_state.api_key_required = True
                        st.error("⚠️ API Key must be entered to execute comparisons. Please set your API key in the sidebar.")
                        st.rerun()
                    else:
                        # Update experiment status to in_progress
                        execute_sql("UPDATE v2_experiments SET status = 'in_progress', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
                        
                        # Execute comparisons
                        run_comparisons_for_experiment(experiment_id)
                        st.rerun()
                
            # Quick status summary
            if n_cases > 0:
                st.write("---")
                st.write("**Quick Status Summary**")
                overall_progress = ((total_tests / n_cases) * 0.5 + (total_comparisons / required_comparisons) * 0.5) if required_comparisons > 0 else (total_tests / n_cases)
                st.progress(overall_progress, text=f"Overall Progress: {overall_progress:.1%}")
                
                if total_tests == n_cases and total_comparisons == required_comparisons:
                    st.success("🎉 Experiment completed! All extractions and comparisons are done.")
                    st.info("💡 Next steps: View results in the Extractions, Comparisons, and Results tabs.")
                elif total_cost >= exp['cost_limit_usd'] * 0.9:
                    st.warning(f"⚠️ Approaching cost limit: ${total_cost:.2f} / ${exp['cost_limit_usd']}")
                    st.info("💡 Consider increasing the cost limit in Settings or analyzing current results.")
        
        with tab4:
            # Extractions tab
            st.subheader("📄 Extracted Legal Tests")
            
            # Query extractions for this experiment
            extractions = execute_sql("""
                SELECT 
                    ee.extraction_id,
                    c.case_name,
                    c.citation,
                    ee.legal_test_name,
                    ee.legal_test_content,
                    ee.extraction_rationale,
                    ee.test_passages,
                    ee.test_novelty,
                    ee.rule_like_score,
                    ee.confidence_score,
                    ee.validation_status,
                    c.decision_url
                FROM v2_experiment_extractions ee
                JOIN v2_cases c ON ee.case_id = c.case_id
                WHERE ee.experiment_id = ?
                ORDER BY c.case_name
            """, (experiment_id,), fetch=True)
            
            if not extractions:
                st.info("No extractions found for this experiment yet. Run extractions first.")
            else:
                # Convert to DataFrame for easy display
                pd = _get_pandas()
                df = pd.DataFrame(extractions, columns=[
                    'extraction_id', 'case_name', 'citation', 'legal_test_name', 
                    'legal_test_content', 'extraction_rationale', 'test_passages', 
                    'test_novelty', 'rule_like_score', 'confidence_score', 
                    'validation_status', 'decision_url'
                ])
                
                # Display table with proper formatting
                st.write(f"**Total Extractions:** {len(df)}")
                
                # Create display columns with proper formatting
                for idx, row in df.iterrows():
                    with st.expander(f"**{row['case_name']}** ({row['citation']})", expanded=False):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write("**Legal Test:**")
                            st.write(row['legal_test_content'])
                            st.write("**Test Location:**")
                            st.write(row['test_passages'] if row['test_passages'] else "Not available")
                            st.write("**Test Novelty:**")
                            st.write(row['test_novelty'] if row['test_novelty'] else "Not available")
                            
                        with col2:
                            st.metric("Rule-like Score", f"{row['rule_like_score']:.2f}" if row['rule_like_score'] else "N/A")
                            st.metric("Confidence", f"{row['confidence_score']:.2f}" if row['confidence_score'] else "N/A")
                            st.write(f"**Status:** {row['validation_status']}")
                            
                            if row['decision_url']:
                                st.link_button("📖 View Case", row['decision_url'])
                            else:
                                st.write("**Case Link:** Not available")
        
        with tab5:
            # Comparisons tab
            st.subheader("⚖️ Pairwise Comparisons")
            
            # Query comparisons for this experiment
            comparisons = execute_sql("""
                SELECT 
                    ec.comparison_id,
                    c1.case_name as case_a_name,
                    c2.case_name as case_b_name,
                    c1.citation as case_a_citation,
                    c2.citation as case_b_citation,
                    ee1.legal_test_content as test_a,
                    ee2.legal_test_content as test_b,
                    winner_case.case_name as winner_case_name,
                    ec.comparison_rationale,
                    ec.confidence_score,
                    ec.human_validated
                FROM v2_experiment_comparisons ec
                JOIN v2_experiment_extractions ee1 ON ec.extraction_id_1 = ee1.extraction_id
                JOIN v2_experiment_extractions ee2 ON ec.extraction_id_2 = ee2.extraction_id
                JOIN v2_cases c1 ON ee1.case_id = c1.case_id
                JOIN v2_cases c2 ON ee2.case_id = c2.case_id
                LEFT JOIN v2_experiment_extractions winner_ext ON ec.winner_id = winner_ext.extraction_id
                LEFT JOIN v2_cases winner_case ON winner_ext.case_id = winner_case.case_id
                WHERE ec.experiment_id = ?
                ORDER BY ec.comparison_date DESC
            """, (experiment_id,), fetch=True)
            
            if not comparisons:
                st.info("No comparisons found for this experiment yet. Run comparisons first.")
            else:
                st.write(f"**Total Comparisons:** {len(comparisons)}")
                
                # Display comparisons
                for idx, comp in enumerate(comparisons):
                    with st.expander(f"**Comparison {idx + 1}:** {comp[1]} vs {comp[2]}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Case A: {comp[1]}** ({comp[3]})")
                            st.write("**Legal Test:**")
                            st.write(comp[5])
                            
                        with col2:
                            st.write(f"**Case B: {comp[2]}** ({comp[4]})")
                            st.write("**Legal Test:**")
                            st.write(comp[6])
                        
                        st.write("---")
                        col3, col4 = st.columns([2, 1])
                        
                        with col3:
                            st.write(f"**Winner:** {comp[7] if comp[7] else 'No winner determined'}")
                            st.write("**Explanation:**")
                            st.write(comp[8] if comp[8] else "No explanation provided")
                            
                        with col4:
                            st.metric("Confidence", f"{comp[9]:.2f}" if comp[9] else "N/A")
                            st.write(f"**Human Validated:** {'Yes' if comp[10] else 'No'}")
        
        with tab6:
            # Results tab
            st.subheader("📊 Analysis Results")
            st.info("🚧 Bradley-Terry analysis and visualization functionality will be implemented here.")
            
            # Placeholder for results
            if total_tests > 0 and total_comparisons > 0:
                st.write("**Analysis Overview:**")
                st.metric("Total Tests Analyzed", total_tests)
                st.metric("Total Comparisons", total_comparisons)
                st.write("Detailed Bradley-Terry scoring and temporal analysis will be available here.")
            else:
                st.write("Complete extractions and comparisons to view analysis results.")
        
        with tab7:
            st.subheader("⚙️ Experiment Settings")
            
            # Configuration Management
            st.write("**Configuration Management**")
            
            # Check if experiment has started execution
            has_extractions = execute_sql("SELECT COUNT(*) FROM v2_experiment_extractions WHERE experiment_id = ?", (experiment_id,), fetch=True)
            has_extractions = has_extractions[0][0] > 0 if has_extractions else False
            
            col1, col2 = st.columns(2)
            
            with col1:
                if has_extractions:
                    st.button("✏️ Edit Configuration", type="secondary", use_container_width=True, disabled=True, 
                             help="Cannot edit configuration after execution has started")
                else:
                    if st.button("✏️ Edit Configuration", type="secondary", use_container_width=True):
                        st.session_state.editing_experiment = experiment_id
                        st.session_state.selected_page = "Create Experiment"
                        st.rerun()
            
            with col2:
                if st.button("📋 Clone Experiment", type="secondary", use_container_width=True):
                    # Set up cloning by copying the experiment data
                    st.session_state.editing_experiment = None
                    st.session_state.clone_from_experiment = experiment_id
                    st.session_state.selected_page = "Create Experiment"
                    st.rerun()
            
            if has_extractions:
                st.info("⚠️ Configuration editing is disabled once execution has started to maintain experiment integrity.")
            
            # Navigation
            st.write("")
            st.write("**Navigation**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Back to Overview", use_container_width=True):
                    st.session_state.selected_page = "Library Overview"
                    st.rerun()
            
            with col2:
                if st.button("📈 View in Comparison", use_container_width=True):
                    st.session_state.selected_page = "Comparison"
                    st.rerun()
            
            # Experiment Status Management
            st.write("")
            st.write("**Status Management**")
            
            current_status = exp['status']
            status_options = ['draft', 'in_progress', 'complete', 'archived']
            status_descriptions = {
                'draft': 'Draft - Experiment is being configured',
                'in_progress': 'In Progress - Experiment is actively running',
                'complete': 'Complete - Experiment has finished successfully',
                'archived': 'Archived - Experiment is stored but not active'
            }
            
            st.write(f"**Current Status:** {status_descriptions[current_status]}")
            
            # Status change buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if current_status == 'draft':
                    st.info("Experiment will automatically transition to 'In Progress' when execution starts.")
                elif current_status == 'in_progress':
                    st.info("Experiment is actively running.")
                elif current_status == 'complete':
                    if st.button("🔄 Revert to In Progress", type="secondary", use_container_width=True):
                        execute_sql("UPDATE v2_experiments SET status = 'in_progress', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
                        st.success("Experiment reverted to in progress!")
                        st.rerun()
                        
            with col2:
                if current_status != 'complete':
                    if st.button("🏁 Mark Complete", type="secondary", use_container_width=True):
                        execute_sql("UPDATE v2_experiments SET status = 'complete', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
                        st.success("Experiment marked as complete!")
                        st.rerun()
                        
            with col3:
                if current_status != 'archived':
                    if st.button("📦 Archive", type="secondary", use_container_width=True):
                        execute_sql("UPDATE v2_experiments SET status = 'archived', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
                        st.success("Experiment archived!")
                        st.rerun()
                        
            # Danger zone
            st.write("")
            st.write("**⚠️ Danger Zone**")
            with st.expander("Advanced Actions", expanded=False):
                st.warning("These actions cannot be undone. Use with caution.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Delete Extractions**")
                    confirm_delete_extractions = st.checkbox(
                        "I confirm I want to permanently delete all extractions for this experiment",
                        key=f"confirm_extractions_{experiment_id}"
                    )
                    if st.button(
                        "🗑️ Delete Extractions", 
                        type="primary" if confirm_delete_extractions else "secondary",
                        disabled=not confirm_delete_extractions,
                        use_container_width=True
                    ):
                        execute_sql("DELETE FROM v2_experiment_extractions WHERE experiment_id = ?", (experiment_id,))
                        
                        # Check if both extractions and comparisons are now empty
                        remaining_extractions = execute_sql("SELECT COUNT(*) FROM v2_experiment_extractions WHERE experiment_id = ?", (experiment_id,), fetch=True)[0][0]
                        remaining_comparisons = execute_sql("SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = ?", (experiment_id,), fetch=True)[0][0]
                        
                        if remaining_extractions == 0 and remaining_comparisons == 0:
                            # Reset experiment status to draft if all work is deleted
                            execute_sql("UPDATE v2_experiments SET status = 'draft' WHERE experiment_id = ?", (experiment_id,))
                            st.success("All extractions deleted! Experiment status reset to draft.")
                        else:
                            st.success("All extractions deleted!")
                        
                        st.cache_data.clear()
                        st.rerun()
                
                with col2:
                    st.write("**Delete Comparisons**")
                    confirm_delete_comparisons = st.checkbox(
                        "I confirm I want to permanently delete all comparisons for this experiment",
                        key=f"confirm_comparisons_{experiment_id}"
                    )
                    if st.button(
                        "🗑️ Delete Comparisons", 
                        type="primary" if confirm_delete_comparisons else "secondary",
                        disabled=not confirm_delete_comparisons,
                        use_container_width=True
                    ):
                        execute_sql("DELETE FROM v2_experiment_comparisons WHERE experiment_id = ?", (experiment_id,))
                        
                        # Check if both extractions and comparisons are now empty
                        remaining_extractions = execute_sql("SELECT COUNT(*) FROM v2_experiment_extractions WHERE experiment_id = ?", (experiment_id,), fetch=True)[0][0]
                        remaining_comparisons = execute_sql("SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = ?", (experiment_id,), fetch=True)[0][0]
                        
                        if remaining_extractions == 0 and remaining_comparisons == 0:
                            # Reset experiment status to draft if all work is deleted
                            execute_sql("UPDATE v2_experiments SET status = 'draft' WHERE experiment_id = ?", (experiment_id,))
                            st.success("All comparisons deleted! Experiment status reset to draft.")
                        else:
                            st.success("All comparisons deleted!")
                        
                        st.cache_data.clear()
                        st.rerun()
                
    except Exception as e:
        st.error(f"Error loading experiment details: {e}")
        st.write("Debug info:")
        st.write(f"Experiment ID: {experiment_id}")
        st.write(f"Error: {str(e)}")

def show():
    """Main dashboard interface"""
    # Initialize database tables
    initialize_experiment_tables()
    
    # Show sidebar navigation
    show_sidebar_navigation()
    
    # Main content area - no redundant title since it's in sidebar
    # Get current page from session state - check both selected_page and page_navigation
    current_page = st.session_state.get('page_navigation') or st.session_state.get('selected_page', 'Cases')
    
    # Render content based on selected page
    if current_page == "Cases":
        show_case_management()
        
    elif current_page == "Library Overview":
        show_experiment_overview()
        
    elif current_page == "Experiment Detail":
        experiment_id = st.session_state.get('selected_experiment')
        if experiment_id:
            show_experiment_detail(experiment_id)
        else:
            st.error("No experiment selected")
            
    elif current_page == "Comparison":
        show_experiment_comparison()
        
    elif current_page == "Create Experiment":
        show_experiment_configuration()
        
    elif current_page == "⚗️ Experiment Execution":
        # Show experiment execution interface
        active_experiment = st.session_state.get('active_experiment')
        if active_experiment:
            experiment_execution.show()
        else:
            st.error("No active experiment selected")
            st.session_state.page_navigation = None  # Reset navigation
            st.session_state.selected_page = "Library Overview"
            st.rerun()