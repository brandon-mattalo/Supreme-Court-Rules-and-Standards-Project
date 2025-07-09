"""
Experiment Execution Interface
Optimized workflow for executing individual experiments
"""

import streamlit as st
import pandas as pd
from config import execute_sql, get_database_connection
from utils.case_management import (
    get_bradley_terry_structure, get_block_summary, 
    generate_bradley_terry_comparison_pairs, calculate_bradley_terry_comparisons
)

# Cache active experiment for 60 seconds
@st.cache_data(ttl=60)
def _get_active_experiment_data(exp_id):
    """Get experiment data from database"""
    exp_data = execute_sql(
        "SELECT * FROM v2_experiments WHERE experiment_id = ?", 
        (exp_id,), 
        fetch=True
    )
    
    if exp_data:
        columns = ['experiment_id', 'name', 'description', 'researcher_name', 'status', 'ai_model', 
                   'temperature', 'top_p', 'top_k', 'max_output_tokens', 
                   'extraction_strategy', 'extraction_prompt', 'comparison_prompt',
                   'system_instruction', 'cost_limit_usd', 'created_date',
                   'modified_date', 'created_by']
        return dict(zip(columns, exp_data[0]))
    
    return None

def get_active_experiment():
    """Get the currently active experiment configuration"""
    if 'active_experiment' not in st.session_state:
        return None
    
    exp_id = st.session_state.active_experiment
    return _get_active_experiment_data(exp_id)

def show_experiment_context():
    """Show current experiment context and quick switching"""
    experiment = get_active_experiment()
    
    if not experiment:
        st.error("No active experiment selected. Please select an experiment from the dashboard.")
        if st.button("📊 Go to Dashboard"):
            st.session_state.page_navigation = "📊 Experiment Dashboard"
            st.rerun()
        return False
    
    # Experiment context bar
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            st.markdown(f"**🧪 Active Experiment:** {experiment['name']}")
            st.caption(f"Model: {experiment['ai_model']} | Strategy: {experiment['extraction_strategy']}")
        
        with col2:
            st.metric("Temperature", f"{experiment['temperature']}")
        
        with col3:
            st.metric("Cost Limit", f"${experiment['cost_limit_usd']}")
        
        with col4:
            if st.button("🔄 Switch"):
                st.session_state.page_navigation = "📊 Experiment Dashboard"
                st.rerun()
    
    st.divider()
    return True

# Cache available experiments for 60 seconds
@st.cache_data(ttl=60)
def _get_available_experiments():
    """Get available experiments from database"""
    return execute_sql("""
        SELECT experiment_id, name, description, status, ai_model 
        FROM v2_experiments 
        WHERE status IN ('draft', 'active')
        ORDER BY modified_date DESC
    """, fetch=True)

def show_experiment_selection():
    """Show experiment selection interface"""
    st.header("🧪 Select Experiment to Execute")
    
    # Get available experiments
    experiments = _get_available_experiments()
    
    if not experiments:
        st.warning("No experiments available for execution.")
        if st.button("📊 Go to Dashboard to Create Experiment"):
            st.session_state.page_navigation = "📊 Experiment Dashboard"
            st.rerun()
        return
    
    # Display experiment selection
    st.subheader("Available Experiments:")
    
    for exp in experiments:
        exp_id, name, description, status, ai_model = exp
        
        with st.container():
            col1, col2, col3 = st.columns([4, 2, 1])
            
            with col1:
                st.write(f"**{name}**")
                st.caption(f"{description or 'No description'}")
            
            with col2:
                st.write(f"Model: {ai_model}")
                status_colors = {
                    'draft': '🟡 Draft',
                    'active': '🟢 Active'
                }
                st.write(f"Status: {status_colors.get(status, status)}")
            
            with col3:
                if st.button("▶️ Execute", key=f"exec_{exp_id}"):
                    st.session_state.active_experiment = exp_id
                    st.rerun()
        
        st.divider()

def show_execution_interface():
    """Show the main execution interface (placeholder for now)"""
    experiment = get_active_experiment()
    
    # Main execution cards (simplified for now)
    st.header("⚗️ Experiment Execution")
    
    # Card 1: Data Loading (Bradley-Terry Aware)
    with st.expander("📁 1. Data Loading", expanded=True):
        st.write("**Current Configuration:**")
        st.write(f"- Model: {experiment['ai_model']}")
        st.write(f"- Temperature: {experiment['temperature']}")
        st.write(f"- Strategy: {experiment['extraction_strategy']}")
        
        # Check Bradley-Terry structure status
        structure = get_bradley_terry_structure()
        block_summary = get_block_summary()
        
        # Updated metrics showing Bradley-Terry structure
        col1, col2, col3 = st.columns(3)
        with col1:
            if structure:
                st.metric("Cases in Structure", len(structure))
            else:
                st.metric("Cases Loaded", "No structure")
        with col2:
            if block_summary:
                st.metric("Bradley-Terry Blocks", block_summary['total_blocks'])
            else:
                st.metric("Ready for Processing", "Structure needed")
        with col3:
            if structure and block_summary:
                comparisons = calculate_bradley_terry_comparisons()
                st.metric("Total Comparisons", comparisons)
                st.caption("Block-optimized design")
            else:
                st.metric("Processing Status", "Awaiting structure")
        
        # Status indicator
        if structure and block_summary:
            st.success("✅ Bradley-Terry structure is ready for experiment execution")
        else:
            st.warning("⚠️ Please generate Bradley-Terry structure in the dashboard before proceeding")
            if st.button("🔄 Refresh Structure Status"):
                st.rerun()
    
    # Card 2: Extraction and Sampling (Placeholder)
    with st.expander("🎯 2. Extraction and Sampling", expanded=False):
        st.info("Extraction and sampling functionality will be adapted from the original interface.")
        st.write("This will use the experiment's configured parameters:")
        st.code(f"""
Extraction Strategy: {experiment['extraction_strategy']}
AI Model: {experiment['ai_model']}
Temperature: {experiment['temperature']}
Max Tokens: {experiment['max_output_tokens']}
""")
    
    # Card 3: Validation (Placeholder)
    with st.expander("✅ 3. Validation", expanded=False):
        st.info("Validation interface will be implemented here.")
    
    # Card 4: Generate Comparisons (Bradley-Terry Structure)
    with st.expander("⚖️ 4. Generate Comparisons", expanded=False):
        st.write("**Bradley-Terry Block Structure Analysis**")
        
        # Check if Bradley-Terry structure exists
        structure = get_bradley_terry_structure()
        
        if structure:
            # Show block summary
            block_summary = get_block_summary()
            if block_summary:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Blocks", block_summary['total_blocks'])
                with col2:
                    st.metric("Core Cases", block_summary['total_core_cases'])
                with col3:
                    st.metric("Bridge Cases", block_summary['total_bridge_cases'])
                with col4:
                    comparisons_needed = calculate_bradley_terry_comparisons()
                    st.metric("Comparisons Needed", comparisons_needed)
            
            # Generate comparison pairs
            if st.button("🎯 Generate Comparison Pairs", type="primary"):
                with st.spinner("Generating optimal comparison pairs..."):
                    pairs, pair_info = generate_bradley_terry_comparison_pairs()
                    
                    if pairs:
                        st.success(f"✅ Generated {len(pairs)} comparison pairs from {len(structure)} cases")
                        
                        # Show sample pairs
                        with st.expander("📋 Sample Comparison Pairs"):
                            sample_pairs = pairs[:10]  # Show first 10 pairs
                            pair_data = []
                            for pair in sample_pairs:
                                info = pair_info[pair]
                                pair_data.append({
                                    'Block': info['block_number'],
                                    'Case 1': info['case1_name'][:30] + "..." if len(info['case1_name']) > 30 else info['case1_name'],
                                    'Role 1': info['case1_role'],
                                    'Case 2': info['case2_name'][:30] + "..." if len(info['case2_name']) > 30 else info['case2_name'],
                                    'Role 2': info['case2_role']
                                })
                            
                            st.dataframe(pd.DataFrame(pair_data), use_container_width=True)
                            st.caption(f"Showing first 10 of {len(pairs)} total pairs")
                    else:
                        st.error("Failed to generate comparison pairs")
            
            # Show current structure summary
            with st.expander("🔍 Block Structure Details"):
                if block_summary:
                    for block_num, block_info in block_summary['blocks_detail'].items():
                        st.write(f"**Block {block_num}:** {block_info['core']} core + {block_info['bridge']} bridge = {block_info['total']} cases")
        else:
            st.warning("⚠️ No Bradley-Terry structure found. Please ensure cases are selected and structure is generated in the dashboard.")
            if st.button("🔄 Check for Structure"):
                st.rerun()
    
    # Card 5: Pairwise Comparisons (Placeholder)
    with st.expander("🔄 5. Pairwise Comparisons", expanded=False):
        st.info("Pairwise comparison interface will be implemented here.")
    
    # Card 6: Analysis (Placeholder)
    with st.expander("📊 6. Analysis", expanded=False):
        st.info("Analysis interface will be implemented here.")
        st.write("Results will be saved to the experiment's result table for cross-experiment comparison.")

def show_progress_tracking():
    """Show experiment execution progress"""
    experiment = get_active_experiment()
    if not experiment:
        return
    
    # Get latest run data
    run_data = execute_sql("""
        SELECT * FROM v2_experiment_runs 
        WHERE experiment_id = ? 
        ORDER BY run_date DESC 
        LIMIT 1
    """, (experiment['experiment_id'],), fetch=True)
    
    if run_data:
        run = run_data[0]
        st.sidebar.subheader("📈 Progress Tracking")
        st.sidebar.metric("Cases Processed", run[4])  # cases_processed
        st.sidebar.metric("Tests Extracted", run[5])  # tests_extracted
        st.sidebar.metric("Comparisons", run[6])      # comparisons_completed
        st.sidebar.metric("Cost So Far", f"${run[7]:.2f}")  # total_cost_usd
        
        # Progress bar
        if experiment['cost_limit_usd'] > 0:
            progress = min(run[7] / experiment['cost_limit_usd'], 1.0)
            st.sidebar.progress(progress, text=f"Budget: {progress:.1%}")

def show():
    """Main experiment execution interface"""
    # Show experiment context
    if not show_experiment_context():
        # No active experiment, show selection interface
        show_experiment_selection()
        return
    
    # Show progress in sidebar
    show_progress_tracking()
    
    # Show execution interface
    show_execution_interface()
    
    # Footer with experiment info
    experiment = get_active_experiment()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Experiment:** {experiment['name']}")
    st.sidebar.markdown(f"**ID:** {experiment['experiment_id']}")
    st.sidebar.markdown(f"**Created:** {experiment['created_date'][:10]}")
    
    # Quick actions
    st.sidebar.subheader("🚀 Quick Actions")
    if st.sidebar.button("📊 View Dashboard"):
        st.session_state.page_navigation = "📊 Experiment Dashboard"
        st.rerun()
    
    if st.sidebar.button("⚙️ Configure Experiment"):
        st.session_state.editing_experiment = experiment['experiment_id']
        st.session_state.page_navigation = "📊 Experiment Dashboard"
        st.rerun()