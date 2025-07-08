"""
Experiment Management Dashboard
Meta-level interface for creating, managing, and comparing experiments
"""

import streamlit as st
from datetime import datetime, timedelta
from config import GEMINI_MODELS, execute_sql, get_database_connection
import json

# Lazy imports for performance
def _get_pandas():
    import pandas as pd
    return pd

def _get_numpy():
    import numpy as np
    return np
from utils.case_management import (
    get_database_counts, load_data_from_parquet, clear_database, 
    get_case_summary, get_available_cases, filter_cases_by_criteria,
    get_experiment_selected_cases, get_available_cases_for_selection,
    add_cases_to_experiments, remove_cases_from_experiments,
    calculate_bradley_terry_comparisons
)

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
            COALESCE(SUM(er.total_cost_usd), 0) as total_cost,
            COALESCE(SUM(er.tests_extracted), 0) as total_tests,
            COALESCE(SUM(er.comparisons_completed), 0) as total_comparisons
        FROM v2_experiments e
        LEFT JOIN v2_experiment_runs er ON e.experiment_id = er.experiment_id
        GROUP BY e.experiment_id, e.name, e.description, e.status, e.ai_model, e.extraction_strategy, e.modified_date
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
def get_cost_calculation_params(n_cases, avg_selected_case_length):
    """Pre-calculate shared cost parameters for all experiment cards"""
    required_comparisons = calculate_bradley_terry_comparisons(n_cases)
    
    # Calculate shared cost parameters
    avg_tokens_selected = avg_selected_case_length / 4
    system_prompt_tokens = 100
    extraction_prompt_tokens = 200
    extracted_test_tokens = 325  # ~250 words * 1.3 tokens/word
    
    # Base cost per extraction for any model
    extraction_input_tokens = avg_tokens_selected + system_prompt_tokens + extraction_prompt_tokens
    
    # Bradley-Terry parameters for display
    block_size = 15  # 12 core + 3 bridge cases per block
    core_cases_per_block = 12
    comparisons_per_block = 105
    
    return {
        'required_comparisons': required_comparisons,
        'extraction_input_tokens': extraction_input_tokens,
        'extracted_test_tokens': extracted_test_tokens,
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
    cost_params = get_cost_calculation_params(n_cases, avg_selected_case_length)
    
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
    
    # Use pre-calculated cost parameters for efficiency
    extraction_input_tokens = cost_params['extraction_input_tokens']
    extracted_test_tokens = cost_params['extracted_test_tokens']
    
    extraction_cost_per_case = (extraction_input_tokens / 1_000_000) * model_pricing['input'] + (extracted_test_tokens / 1_000_000) * model_pricing['output']
    
    # Calculate estimates
    remaining_extractions = max(0, n_cases - int(exp['total_tests'] or 0))
    remaining_comparisons = max(0, required_comparisons - int(exp['total_comparisons'] or 0))
    
    extraction_cost_estimate = remaining_extractions * extraction_cost_per_case
    sample_total_cost = (n_cases * extraction_cost_per_case) + (required_comparisons * 0.0003)  # Rough comparison cost
    
    # Status and progress
    status_colors = {
        'draft': '🟡 Draft',
        'active': '🟢 Active', 
        'completed': '🔵 Completed',
        'archived': '⚫ Archived'
    }
    
    # Use Streamlit's built-in container with visual separation
    with st.container(border=True):
        # Card content
        st.subheader(f"🧪 #{exp['experiment_id']} {exp['name']}")
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
        
        # Actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⚙️ Configure", key=f"config_{exp['experiment_id']}", use_container_width=True):
                st.session_state.editing_experiment = exp['experiment_id']
                st.rerun()
        
        with col2:
            if st.button("▶️ Execute", key=f"execute_{exp['experiment_id']}", use_container_width=True):
                st.session_state.active_experiment = exp['experiment_id']
                st.session_state.page_navigation = "⚗️ Experiment Execution"
                st.rerun()
        
        with col3:
            if st.button("📊 Details", key=f"details_{exp['experiment_id']}", use_container_width=True):
                st.session_state.selected_page = "Experiment Detail"
                st.session_state.selected_experiment = exp['experiment_id']
                st.rerun()

def show_experiment_configuration():
    """Show experiment configuration interface"""
    st.header("⚙️ Experiment Configuration")
    
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
    
    with st.form("experiment_config"):
        # Basic Information
        st.subheader("📝 Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Experiment Name", value=exp['name'])
            researcher_name = st.text_input("Researcher's Name", value=exp['researcher_name'])
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
            extraction_strategy = st.selectbox(
                "Extraction Strategy", 
                ['single_test', 'multi_test', 'full_text_comparison'],
                index=['single_test', 'multi_test', 'full_text_comparison'].index(exp['extraction_strategy']),
                help="single_test: Extract one primary legal test per case | multi_test: Extract multiple tests if present | full_text_comparison: Compare entire case texts for patterns"
            )
        
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
                base_name = name if name and name.strip() else "Unnamed_Experiment"
                new_name = f"{base_name}_copy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if save_experiment(None, new_name, description, researcher_name, status, ai_model, temperature, 
                                  top_p, top_k, max_tokens, extraction_strategy, extraction_prompt,
                                  comparison_prompt, system_instruction, cost_limit):
                    st.success(f"Experiment cloned as '{new_name}'!")
                    st.session_state.editing_experiment = None
                    st.rerun()
        
        with col3:
            if st.form_submit_button("❌ Cancel"):
                st.session_state.editing_experiment = None
                st.rerun()
        
        if submitted:
            # Validate required fields
            if not name or not name.strip():
                st.error("Experiment name is required!")
            else:
                if save_experiment(editing_id, name, description, researcher_name, status, ai_model, temperature,
                                 top_p, top_k, max_tokens, extraction_strategy, extraction_prompt,
                                 comparison_prompt, system_instruction, cost_limit):
                    st.success("Experiment saved successfully!")
                    st.session_state.editing_experiment = None
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
                    name = ?, description = ?, researcher_name = ?, status = ?, ai_model = ?, temperature = ?,
                    top_p = ?, top_k = ?, max_output_tokens = ?, extraction_strategy = ?,
                    extraction_prompt = ?, comparison_prompt = ?, system_instruction = ?,
                    cost_limit_usd = ?, modified_date = CURRENT_TIMESTAMP
                WHERE experiment_id = ?
            """, (name, description, researcher_name, status, ai_model, temperature, top_p, top_k, max_tokens,
                  extraction_strategy, extraction_prompt, comparison_prompt, system_instruction,
                  cost_limit, experiment_id))
        else:
            # Create new experiment
            execute_sql("""
                INSERT INTO v2_experiments (name, description, researcher_name, status, ai_model, temperature, top_p,
                                       top_k, max_output_tokens, extraction_strategy, extraction_prompt,
                                       comparison_prompt, system_instruction, cost_limit_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, description, researcher_name, status, ai_model, temperature, top_p, top_k, max_tokens,
                  extraction_strategy, extraction_prompt, comparison_prompt, system_instruction, cost_limit))
        
        # Clear caches after modifying experiments
        get_experiments_list.clear()
        _get_experiment_detail.clear()
        
        return True
    except Exception as e:
        st.error(f"Error saving experiment: {e}")
        return False

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
                if st.button("Clear All Data", disabled=not is_admin, type="secondary"):
                    if st.checkbox("I understand this will delete ALL data", key="confirm_clear"):
                        if clear_database():
                            st.rerun()
    
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
                    
                    num_to_select = st.number_input(
                        f"Number of cases to randomly select",
                        min_value=1, 
                        max_value=min(available_count, 100), 
                        value=min(10, available_count),
                        help=f"{available_count} cases available with current filters"
                    )
            
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
            
            # Add cases button
            if st.button("🎲 Randomly Select and Add Cases", type="primary"):
                if num_to_select > 0:
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
                    'active': '🟢', 
                    'completed': '🔵',
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
        
        st.title(f"🧪 Experiment #{exp['experiment_id']}: {exp['name']}")
        
        # Status with color
        status_colors = {
            'draft': '🟡 Draft',
            'active': '🟢 Active', 
            'completed': '🔵 Completed',
            'archived': '⚫ Archived'
        }
        st.markdown(f"**Status:** {status_colors.get(exp['status'], exp['status'])}")
        
        # Get shared data for cost calculations
        try:
            selected_cases_count = execute_sql("SELECT COUNT(*) FROM v2_experiment_selected_cases", fetch=True)[0][0]
            total_cases_count = execute_sql("SELECT COUNT(*) FROM v2_cases", fetch=True)[0][0]
            
            if selected_cases_count > 0:
                selected_lengths = execute_sql("""
                    SELECT AVG(c.case_length) 
                    FROM v2_cases c 
                    JOIN v2_experiment_selected_cases s ON c.case_id = s.case_id
                    WHERE c.case_length IS NOT NULL
                """, fetch=True)
                avg_selected_case_length = float(selected_lengths[0][0]) if selected_lengths and selected_lengths[0][0] else 52646.0
            else:
                avg_selected_case_length = 52646.0
                
            all_lengths = execute_sql("SELECT AVG(case_length) FROM v2_cases WHERE case_length IS NOT NULL", fetch=True)
            avg_all_case_length = float(all_lengths[0][0]) if all_lengths and all_lengths[0][0] else 52646.0
        except:
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
        
        # Get run statistics
        runs_data = execute_sql("""
            SELECT COUNT(*) as run_count,
                   SUM(tests_extracted) as total_tests,
                   SUM(comparisons_completed) as total_comparisons,
                   SUM(total_cost_usd) as total_cost
            FROM v2_experiment_runs 
            WHERE experiment_id = ?
        """, (experiment_id,), fetch=True)
        
        if runs_data and runs_data[0]:
            stats = runs_data[0]
            total_tests = stats[1] or 0
            total_comparisons = stats[2] or 0
            total_cost = stats[3] or 0
        else:
            total_tests = 0
            total_comparisons = 0
            total_cost = 0
        
        # Layout in tabs for organization
        tab1, tab2, tab3 = st.tabs(["📋 Configuration", "📊 Progress & Stats", "🎯 Actions"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Information")
                st.write(f"**Description:** {exp['description'] or 'No description'}")
                st.write(f"**Researcher:** {exp.get('researcher_name', 'Not specified')}")
                st.write(f"**Created:** {exp['created_date'][:10] if exp['created_date'] else 'Unknown'}")
                st.write(f"**Modified:** {exp['modified_date'][:10] if exp['modified_date'] else 'Unknown'}")
            
            with col2:
                st.subheader("AI Configuration")
                st.write(f"**Model:** {exp['ai_model']}")
                st.write(f"**Temperature:** {exp['temperature']}")
                st.write(f"**Top P:** {exp.get('top_p', 1.0)}")
                st.write(f"**Top K:** {exp.get('top_k', 40)}")
                st.write(f"**Max Tokens:** {exp.get('max_output_tokens', 8192)}")
                st.write(f"**Strategy:** {exp['extraction_strategy']}")
                st.write(f"**Cost Limit:** ${exp['cost_limit_usd']}")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
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
                st.subheader("Sample Information")
                st.metric("Sample Size", f"{n_cases} cases")
                st.metric("Total Database", f"{total_cases_count:,} cases")
                st.metric("Required Comparisons", f"{required_comparisons:,}")
                
                # Show comparison strategy
                if n_cases <= block_size:
                    comparison_strategy = "Full pairwise"
                else:
                    blocks_needed = (n_cases + core_cases_per_block - 1) // core_cases_per_block
                    comparison_strategy = f"Bradley-Terry ({blocks_needed} blocks)"
                
                st.write(f"**Comparison Strategy:** {comparison_strategy}")
        
        with tab3:
            st.subheader("🎯 Available Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("⚙️ Edit Configuration", type="secondary", use_container_width=True):
                    st.session_state.editing_experiment = experiment_id
                    st.session_state.selected_page = "Create Experiment"
                    st.rerun()
            
            with col2:
                if st.button("▶️ Execute Experiment", type="primary", use_container_width=True):
                    st.session_state.active_experiment = experiment_id
                    st.session_state.page_navigation = "⚗️ Experiment Execution"
                    st.rerun()
            
            with col3:
                if st.button("📈 View in Comparison", type="secondary", use_container_width=True):
                    st.session_state.selected_page = "Comparison"
                    st.rerun()
            
            st.write("")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Back to Overview", use_container_width=True):
                    st.session_state.selected_page = "Library Overview"
                    st.rerun()
            
            with col2:
                if st.button("📋 Clone Experiment", use_container_width=True):
                    # Set up cloning by copying the experiment data
                    st.session_state.editing_experiment = None
                    st.session_state.clone_from_experiment = experiment_id
                    st.session_state.selected_page = "Create Experiment"
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
    
    # Main content area
    st.title("🧪 Legal Research Experimentation Platform")
    
    # Get current page from session state
    current_page = st.session_state.get('selected_page', 'Cases')
    
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