"""
Legal Research Experimentation Platform - Version 2.0
Main application entry point with multi-page navigation
"""

import streamlit as st

# Set page config first to optimize rendering
st.set_page_config(
    page_title="SCC Research Platform v2.0",
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page imports (lazy loading)
from pages import dashboard

def setup_sidebar():
    """Setup the sidebar with minimal header"""
    # Simple header
    st.sidebar.markdown("**🏛️ Legal Research Platform**")

def main():
    """Main application entry point"""
    # Setup sidebar
    setup_sidebar()
    
    # Show dashboard
    dashboard.show()

if __name__ == "__main__":
    main()