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
from utils.case_management import (
    get_database_counts, load_data_from_parquet, clear_database, 
    get_case_summary, get_available_cases, filter_cases_by_criteria
)

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
    st.markdown("*Shared case database used across all experiments*")
    
    # Get case summary
    summary = get_case_summary()
    
    if summary:
        # Overview metrics
        st.subheader("📊 Database Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cases", summary['total_cases'])
        with col2:
            st.metric("Extracted Tests", summary['total_tests'])
        with col3:
            st.metric("Validated Tests", summary['validated_tests'])
        with col4:
            st.metric("Comparisons Made", summary['total_comparisons'])
        
        # Additional statistics
        if summary['year_range'][0]:
            year_span = f"{summary['year_range'][0]} - {summary['year_range'][1]}"
            st.metric("Year Range", year_span, f"{summary['year_range'][2]} years")
        
        # Top areas of law
        if summary['top_areas']:
            st.subheader("📈 Top Areas of Law")
            areas_df = pd.DataFrame(summary['top_areas'], columns=['Area of Law', 'Case Count'])
            st.bar_chart(areas_df.set_index('Area of Law'))
    
    st.divider()
    
    # Data Loading Interface
    st.subheader("📁 Data Loading")
    
    # Admin password protection
    admin_password = st.text_input("🔐 Admin Password (required for data operations)", 
                                 type="password", key="admin_password_dashboard")
    is_admin = admin_password == "parquet2040"
    
    if not is_admin:
        st.warning("⚠️ Admin password required to access data loading and clearing functions.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Upload Data")
            uploaded_file = st.file_uploader("Choose a Parquet file", type="parquet", 
                                           disabled=not is_admin)
            if uploaded_file is not None and is_admin:
                if st.button("🚀 Load Data", type="primary"):
                    with st.spinner("Loading data..."):
                        load_data_from_parquet(uploaded_file)
                        st.rerun()  # Refresh to update metrics
        
        with col2:
            st.subheader("🗑️ Database Management")
            if st.button("Clear All Data", disabled=not is_admin, type="secondary"):
                if st.checkbox("I understand this will delete ALL data", key="confirm_clear"):
                    if clear_database():
                        st.rerun()
    
    st.divider()
    
    # Case Filtering and Sampling
    st.subheader("🎯 Case Filtering & Sampling")
    
    available_cases = get_available_cases()
    
    if not available_cases.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Year range filter
            min_year = int(available_cases['decision_year'].min())
            max_year = int(available_cases['decision_year'].max())
            year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year))
            
            # Area of law filter
            unique_areas = available_cases['area_of_law'].dropna().unique()
            selected_areas = st.multiselect("Areas of Law", unique_areas, default=unique_areas[:5])
            
            # Sample size
            sample_size = st.number_input("Sample Size (0 = all cases)", 
                                        min_value=0, max_value=len(available_cases), 
                                        value=min(100, len(available_cases)))
        
        with col2:
            st.write("**Filter Preview:**")
            
            if st.button("🔍 Apply Filters"):
                filtered_cases = filter_cases_by_criteria(
                    year_range=year_range,
                    areas=selected_areas if selected_areas else None,
                    sample_size=sample_size if sample_size > 0 else None
                )
                
                st.session_state.filtered_cases = filtered_cases
                st.success(f"Filtered to {len(filtered_cases)} cases")
        
        # Show filtered results
        if 'filtered_cases' in st.session_state and not st.session_state.filtered_cases.empty:
            st.subheader("📋 Filtered Cases")
            
            # Add export functionality
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{len(st.session_state.filtered_cases)} cases selected**")
            with col2:
                csv = st.session_state.filtered_cases.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv,
                    file_name=f"filtered_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Paginated display
            items_per_page = st.selectbox("Cases per page:", [10, 25, 50], index=1)
            total_pages = (len(st.session_state.filtered_cases) + items_per_page - 1) // items_per_page
            
            if total_pages > 1:
                current_page = st.selectbox("Page:", range(1, total_pages + 1)) - 1
            else:
                current_page = 0
            
            start_idx = current_page * items_per_page
            end_idx = start_idx + items_per_page
            page_data = st.session_state.filtered_cases.iloc[start_idx:end_idx]
            
            # Display cases
            for idx, case in page_data.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.write(f"**{case['case_name']}**")
                        st.caption(f"Citation: {case['citation']}")
                    
                    with col2:
                        st.write(f"Year: {case['decision_year']}")
                        st.write(f"Area: {case['area_of_law']}")
                    
                    with col3:
                        st.write(f"ID: {case['case_id']}")
                
                st.divider()
    else:
        st.info("No cases available. Upload data to begin case management.")

def show_experiment_comparison():
    """Show cross-experiment comparison interface"""
    st.header("📈 Experiment Comparison")
    st.markdown("*Compare methodology effectiveness across different experimental configurations*")
    
    # Get experiments with results
    experiments_with_results = execute_sql("""
        SELECT DISTINCT 
            e.experiment_id, 
            e.name, 
            e.ai_model, 
            e.temperature, 
            e.extraction_strategy,
            COUNT(er.run_id) as run_count,
            SUM(er.tests_extracted) as total_tests,
            SUM(er.comparisons_completed) as total_comparisons,
            AVG(er.total_cost_usd) as avg_cost,
            MAX(er.run_date) as last_run
        FROM experiments e
        LEFT JOIN experiment_runs er ON e.experiment_id = er.experiment_id
        WHERE e.status IN ('active', 'completed') 
        GROUP BY e.experiment_id, e.name, e.ai_model, e.temperature, e.extraction_strategy
        HAVING run_count > 0
        ORDER BY e.modified_date DESC
    """, fetch=True)
    
    if not experiments_with_results:
        st.info("No experiments with results available for comparison. Run some experiments first!")
        return
    
    # Convert to DataFrame for easier handling
    columns = ['experiment_id', 'name', 'ai_model', 'temperature', 'extraction_strategy', 
               'run_count', 'total_tests', 'total_comparisons', 'avg_cost', 'last_run']
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