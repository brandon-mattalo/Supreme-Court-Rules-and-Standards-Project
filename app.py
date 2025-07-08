"""
Legal Research Experimentation Platform - Version 2.0
Main application entry point with multi-page navigation
"""

import streamlit as st
from config import load_api_key, save_api_key, delete_api_key

# Set page config first to optimize rendering
st.set_page_config(
    page_title="SCC Research Platform v2.0",
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page imports (lazy loading)
from pages import dashboard

def show_settings_popup():
    """Show settings popup modal"""
    # Load existing API key
    current_api_key = load_api_key()
    
    st.subheader("⚙️ Application Settings")
    
    # API Key Management Section
    st.markdown("### 🔑 API Key Management")
    
    if current_api_key:
        st.success("✅ API Key configured")
        if st.button("🗑️ Delete API Key", type="secondary"):
            if delete_api_key():
                st.success("API key deleted!")
                st.rerun()
    else:
        st.warning("⚠️ No API Key found")
    
    # API Key input
    new_api_key = st.text_input(
        "Enter Gemini API Key:",
        type="password",
        value=current_api_key if current_api_key else "",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save API Key", type="primary") and new_api_key:
            if save_api_key(new_api_key):
                st.success("API key saved!")
                st.session_state.gemini_api_key = new_api_key
                st.rerun()
    
    with col2:
        if st.button("❌ Close Settings"):
            st.session_state.show_settings = False
            st.rerun()

def setup_sidebar():
    """Setup the sidebar with minimal header and settings"""
    # Load existing API key and store in session state if available
    current_api_key = load_api_key()
    if current_api_key and 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = current_api_key
    
    # Compact header with settings
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        st.markdown("**🏛️ Legal Research Platform**")
    with col2:
        # Check if API key is missing
        api_key_missing = not current_api_key
        
        # Settings button with custom styling
        help_text = "Gemini API key not set" if api_key_missing else "Settings"
        button_type = "secondary" if api_key_missing else None
        
        # Create a container with unique class for targeted styling
        if api_key_missing:
            st.markdown("""
            <style>
            .settings-button-warning button {
                background-color: #fff3cd !important;
                background: #fff3cd !important;
                border-color: #ffeaa7 !important;
                color: #856404 !important;
            }
            .settings-button-warning button:hover {
                background-color: #ffeaa7 !important;
                background: #ffeaa7 !important;
                border-color: #ffc107 !important;
                color: #856404 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Wrap button in a div with unique class
            st.markdown('<div class="settings-button-warning">', unsafe_allow_html=True)
            settings_clicked = st.button("⚙️", help=help_text, key="settings_btn", type=button_type)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            settings_clicked = st.button("⚙️", help=help_text, key="settings_btn", type=button_type)
        
        if settings_clicked:
            st.session_state.show_settings = True
            st.rerun()

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Legal Research Platform v2.0",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize settings state
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    
    # Setup sidebar
    setup_sidebar()
    
    # Show settings popup if requested
    if st.session_state.show_settings:
        with st.container():
            show_settings_popup()
    else:
        # Show dashboard
        dashboard.show()

if __name__ == "__main__":
    main()