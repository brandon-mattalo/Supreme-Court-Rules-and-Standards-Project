"""
Legal Research Experimentation Platform - Version 2.0
Main application entry point with multi-page navigation
"""

import streamlit as st
from config import load_api_key, save_api_key, delete_api_key, list_available_models, DEFAULT_MODEL

# Page imports
from pages import dashboard, experiment_execution

def setup_sidebar():
    """Setup the sidebar with API key management and navigation"""
    st.sidebar.title("🏛️ Legal Research Platform v2.0")
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Select Interface:",
        ["📊 Experiment Dashboard", "⚗️ Experiment Execution"],
        key="page_navigation"
    )
    
    st.sidebar.markdown("---")
    
    # API Key Management
    st.sidebar.markdown("### 🔑 API Key Management")
    
    # Load existing API key
    current_api_key = load_api_key()
    
    if current_api_key:
        st.sidebar.success("✅ API Key configured")
        if st.sidebar.button("🗑️ Delete API Key"):
            if delete_api_key():
                st.sidebar.success("API key deleted!")
                st.rerun()
    else:
        st.sidebar.warning("⚠️ No API Key found")
    
    # API Key input
    with st.sidebar.expander("Configure API Key"):
        new_api_key = st.text_input(
            "Enter Gemini API Key:",
            type="password",
            value=current_api_key if current_api_key else "",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        if st.button("💾 Save API Key") and new_api_key:
            if save_api_key(new_api_key):
                st.success("API key saved!")
                st.session_state.gemini_api_key = new_api_key
                st.rerun()
    
    # Model Selection
    st.sidebar.markdown("### 🤖 Default Model Selection")
    
    # Store API key in session state if available
    if current_api_key and 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = current_api_key
    
    available_models = list_available_models()
    
    if 'selected_gemini_model' not in st.session_state:
        st.session_state.selected_gemini_model = DEFAULT_MODEL
    
    selected_model = st.sidebar.selectbox(
        "Choose Gemini Model:",
        available_models,
        index=available_models.index(st.session_state.selected_gemini_model) if st.session_state.selected_gemini_model in available_models else 0,
        help="Select the Gemini model for AI operations"
    )
    
    st.session_state.selected_gemini_model = selected_model
    
    return page

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Legal Research Platform v2.0",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Setup sidebar and get selected page
    current_page = setup_sidebar()
    
    # Route to appropriate page
    if current_page == "📊 Experiment Dashboard":
        dashboard.show()
    elif current_page == "⚗️ Experiment Execution":
        experiment_execution.show()

if __name__ == "__main__":
    main()