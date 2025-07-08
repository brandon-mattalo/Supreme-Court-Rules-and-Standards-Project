"""
Experiment Management Dashboard
Meta-level interface for creating, managing, and comparing experiments
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import GEMINI_MODELS, execute_sql, get_database_connection
import json

def initialize_experiment_tables():
    """Initialize database tables for experiment management"""
    
    # Experiments table
    execute_sql('''
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'draft',
            ai_model TEXT DEFAULT 'gemini-2.5-pro',
            temperature REAL DEFAULT 0.0,
            top_p REAL DEFAULT 1.0,
            top_k INTEGER DEFAULT 1,
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
    
    # Experiment runs table
    execute_sql('''
        CREATE TABLE IF NOT EXISTS experiment_runs (
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
            FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
        );
    ''')
    
    # Experiment results summary table
    execute_sql('''
        CREATE TABLE IF NOT EXISTS experiment_results (
            result_id INTEGER PRIMARY KEY,
            experiment_id INTEGER,
            metric_type TEXT,
            metric_value REAL,
            bt_statistics_json TEXT,
            regression_results_json TEXT,
            confidence_scores_json TEXT,
            calculated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
        );
    ''')
    
    # Add indexes for performance
    execute_sql('CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments (status);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_experiment_runs_experiment_id ON experiment_runs (experiment_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_experiment_results_experiment_id ON experiment_results (experiment_id);')

def show_experiment_overview():
    """Display overview of all experiments"""
    st.header("📊 Experiment Overview")
    
    # Get experiment data
    experiments_data = execute_sql("""
        SELECT 
            e.*,
            COUNT(er.run_id) as total_runs,
            MAX(er.run_date) as last_run_date,
            SUM(er.total_cost_usd) as total_cost,
            SUM(er.tests_extracted) as total_tests,
            SUM(er.comparisons_completed) as total_comparisons
        FROM experiments e
        LEFT JOIN experiment_runs er ON e.experiment_id = er.experiment_id
        GROUP BY e.experiment_id
        ORDER BY e.modified_date DESC
    """, fetch=True)
    
    if not experiments_data:
        st.info("No experiments found. Create your first experiment below!")
        return
    
    # Convert to DataFrame
    columns = ['experiment_id', 'name', 'description', 'status', 'ai_model', 'temperature', 
               'top_p', 'top_k', 'max_output_tokens', 'extraction_strategy', 'extraction_prompt',
               'comparison_prompt', 'system_instruction', 'cost_limit_usd', 'created_date',
               'modified_date', 'created_by', 'total_runs', 'last_run_date', 'total_cost',
               'total_tests', 'total_comparisons']
    
    df = pd.DataFrame(experiments_data, columns=columns)
    
    # Display experiment cards
    for _, exp in df.iterrows():
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.subheader(f"🧪 {exp['name']}")
                st.write(f"**Description:** {exp['description'] or 'No description'}")
                st.write(f"**Model:** {exp['ai_model']} (temp: {exp['temperature']})")
                st.write(f"**Strategy:** {exp['extraction_strategy']}")
            
            with col2:
                # Status badge
                status_colors = {
                    'draft': '🟡 Draft',
                    'active': '🟢 Active', 
                    'completed': '🔵 Completed',
                    'archived': '⚫ Archived'
                }
                st.metric("Status", status_colors.get(exp['status'], exp['status']))
                st.metric("Total Runs", int(exp['total_runs'] or 0))
            
            with col3:
                st.metric("Tests Extracted", int(exp['total_tests'] or 0))
                st.metric("Comparisons", int(exp['total_comparisons'] or 0))
            
            with col4:
                st.metric("Total Cost", f"${exp['total_cost'] or 0:.2f}")
                if st.button("⚙️ Configure", key=f"config_{exp['experiment_id']}"):
                    st.session_state.editing_experiment = exp['experiment_id']
                    st.rerun()
                if st.button("▶️ Execute", key=f"execute_{exp['experiment_id']}"):
                    st.session_state.active_experiment = exp['experiment_id']
                    st.session_state.page_navigation = "⚗️ Experiment Execution"
                    st.rerun()
        
        st.divider()

def show_experiment_configuration():
    """Show experiment configuration interface"""
    st.header("⚙️ Experiment Configuration")
    
    # Check if we're editing an existing experiment
    editing_id = st.session_state.get('editing_experiment')
    
    if editing_id:
        # Load existing experiment
        exp_data = execute_sql(
            "SELECT * FROM experiments WHERE experiment_id = ?", 
            (editing_id,), 
            fetch=True
        )
        if exp_data:
            exp = dict(zip(['experiment_id', 'name', 'description', 'status', 'ai_model', 
                           'temperature', 'top_p', 'top_k', 'max_output_tokens', 
                           'extraction_strategy', 'extraction_prompt', 'comparison_prompt',
                           'system_instruction', 'cost_limit_usd', 'created_date',
                           'modified_date', 'created_by'], exp_data[0]))
        else:
            st.error("Experiment not found!")
            return
    else:
        # Default values for new experiment
        exp = {
            'name': '',
            'description': '',
            'status': 'draft',
            'ai_model': 'gemini-2.5-pro',
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': 1,
            'max_output_tokens': 8192,
            'extraction_strategy': 'single_test',
            'extraction_prompt': '',
            'comparison_prompt': '',
            'system_instruction': 'You are a helpful assistant that helps legal researchers analyze legal texts.',
            'cost_limit_usd': 100.0
        }
    
    with st.form("experiment_config"):
        # Basic Information
        st.subheader("📝 Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Experiment Name", value=exp['name'], required=True)
            status = st.selectbox("Status", ['draft', 'active', 'completed', 'archived'], 
                                index=['draft', 'active', 'completed', 'archived'].index(exp['status']))
        
        with col2:
            description = st.text_area("Description", value=exp['description'])
            cost_limit = st.number_input("Cost Limit (USD)", min_value=0.0, value=float(exp['cost_limit_usd']))
        
        # AI Model Configuration
        st.subheader("🤖 AI Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            ai_model = st.selectbox("AI Model", list(GEMINI_MODELS.keys()), 
                                  index=list(GEMINI_MODELS.keys()).index(exp['ai_model']))
            temperature = st.slider("Temperature", 0.0, 2.0, float(exp['temperature']), step=0.1)
            top_p = st.slider("Top P", 0.0, 1.0, float(exp['top_p']), step=0.1)
        
        with col2:
            top_k = st.number_input("Top K", min_value=1, value=int(exp['top_k']))
            max_tokens = st.number_input("Max Output Tokens", min_value=1, max_value=16384, 
                                       value=int(exp['max_output_tokens']))
            extraction_strategy = st.selectbox("Extraction Strategy", 
                                             ['single_test', 'multi_test', 'full_text_comparison'],
                                             index=['single_test', 'multi_test', 'full_text_comparison'].index(exp['extraction_strategy']))
        
        # Prompts Configuration
        st.subheader("📝 Prompts Configuration")
        
        system_instruction = st.text_area("System Instruction", value=exp['system_instruction'], height=100)
        
        col1, col2 = st.columns(2)
        with col1:
            extraction_prompt = st.text_area("Extraction Prompt", value=exp['extraction_prompt'], height=200,
                                           help="Custom prompt for legal test extraction (leave empty to use default)")
        
        with col2:
            comparison_prompt = st.text_area("Comparison Prompt", value=exp['comparison_prompt'], height=200,
                                           help="Custom prompt for test comparison (leave empty to use default)")
        
        # Form submission
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            submitted = st.form_submit_button("💾 Save Experiment", type="primary")
        
        with col2:
            if editing_id and st.form_submit_button("📋 Clone Experiment"):
                # Create a copy of the experiment
                new_name = f"{name}_copy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                save_experiment(None, new_name, description, status, ai_model, temperature, 
                               top_p, top_k, max_tokens, extraction_strategy, extraction_prompt,
                               comparison_prompt, system_instruction, cost_limit)
                st.success(f"Experiment cloned as '{new_name}'!")
                st.session_state.editing_experiment = None
                st.rerun()
        
        with col3:
            if st.form_submit_button("❌ Cancel"):
                st.session_state.editing_experiment = None
                st.rerun()
        
        if submitted:
            if save_experiment(editing_id, name, description, status, ai_model, temperature,
                             top_p, top_k, max_tokens, extraction_strategy, extraction_prompt,
                             comparison_prompt, system_instruction, cost_limit):
                st.success("Experiment saved successfully!")
                st.session_state.editing_experiment = None
                st.rerun()

def save_experiment(experiment_id, name, description, status, ai_model, temperature, top_p, 
                   top_k, max_tokens, extraction_strategy, extraction_prompt, 
                   comparison_prompt, system_instruction, cost_limit):
    """Save experiment configuration to database"""
    try:
        if experiment_id:
            # Update existing experiment
            execute_sql("""
                UPDATE experiments SET 
                    name = ?, description = ?, status = ?, ai_model = ?, temperature = ?,
                    top_p = ?, top_k = ?, max_output_tokens = ?, extraction_strategy = ?,
                    extraction_prompt = ?, comparison_prompt = ?, system_instruction = ?,
                    cost_limit_usd = ?, modified_date = CURRENT_TIMESTAMP
                WHERE experiment_id = ?
            """, (name, description, status, ai_model, temperature, top_p, top_k, max_tokens,
                  extraction_strategy, extraction_prompt, comparison_prompt, system_instruction,
                  cost_limit, experiment_id))
        else:
            # Create new experiment
            execute_sql("""
                INSERT INTO experiments (name, description, status, ai_model, temperature, top_p,
                                       top_k, max_output_tokens, extraction_strategy, extraction_prompt,
                                       comparison_prompt, system_instruction, cost_limit_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, description, status, ai_model, temperature, top_p, top_k, max_tokens,
                  extraction_strategy, extraction_prompt, comparison_prompt, system_instruction, cost_limit))
        
        return True
    except Exception as e:
        st.error(f"Error saving experiment: {e}")
        return False

def show_case_management():
    """Show case management interface"""
    st.header("📚 Case Management")
    st.info("Case extraction and sampling tools will be moved here in the next phase. "
            "This will be shared across all experiments since the same cases are used.")
    
    # Placeholder for case management interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Case Statistics")
        # This will be implemented when we move case management from the execution interface
        st.metric("Total Cases", "TBD")
        st.metric("Processed Cases", "TBD")
        st.metric("Available Cases", "TBD")
    
    with col2:
        st.subheader("🎯 Sampling Configuration")
        st.info("Sampling configuration will be implemented here to ensure consistent case selection across experiments.")

def show_experiment_comparison():
    """Show cross-experiment comparison interface"""
    st.header("📈 Experiment Comparison")
    st.info("Cross-experiment analytics and comparison tools will be implemented here. "
            "This will include Bradley-Terry result comparisons, model performance metrics, "
            "and statistical significance testing.")
    
    # Placeholder for comparison interface
    st.subheader("🎯 Select Experiments to Compare")
    
    experiments = execute_sql("SELECT experiment_id, name FROM experiments WHERE status != 'draft'", fetch=True)
    
    if experiments:
        exp_options = {f"{exp[1]} (ID: {exp[0]})": exp[0] for exp in experiments}
        selected_experiments = st.multiselect("Choose experiments to compare:", exp_options.keys())
        
        if len(selected_experiments) >= 2:
            st.success(f"Selected {len(selected_experiments)} experiments for comparison")
            # Placeholder for comparison results
            st.info("Comparison analytics will be displayed here")
        else:
            st.warning("Select at least 2 experiments to enable comparison")
    else:
        st.info("No active experiments available for comparison")

def show():
    """Main dashboard interface"""
    # Initialize database tables
    initialize_experiment_tables()
    
    # Main dashboard layout
    st.title("🧪 Legal Research Experimentation Platform")
    st.markdown("*Meta-level experiment management and comparison interface*")
    
    # Tab navigation for dashboard sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⚙️ Configuration", "📚 Cases", "📈 Comparison"])
    
    with tab1:
        show_experiment_overview()
        
        # Quick create experiment button
        if st.button("➕ Create New Experiment", type="primary"):
            st.session_state.editing_experiment = None  # Clear any existing edit state
            st.session_state.dashboard_tab = "Configuration"
            st.rerun()
    
    with tab2:
        show_experiment_configuration()
    
    with tab3:
        show_case_management()
    
    with tab4:
        show_experiment_comparison()