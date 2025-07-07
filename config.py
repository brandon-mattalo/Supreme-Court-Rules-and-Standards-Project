
import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import json

load_dotenv()

DB_NAME = "/Users/brandon/My Drive/Learning/Coding/SCC Research/scc_analysis_project/parquet/scc_cases.db"
API_KEY_FILE = "/Users/brandon/My Drive/Learning/Coding/SCC Research/scc_analysis_project/.api_key.json"

# Updated pricing for current Gemini models (2025)
GEMINI_MODELS = {
    'gemini-2.5-pro': {'input': 1.25, 'output': 10.0, 'input_high_volume': 2.5, 'output_high_volume': 15.0},
    'gemini-2.5-flash': {'input': 0.10, 'output': 0.40},
    'gemini-2.5-flash-lite': {'input': 0.075, 'output': 0.30},
    'gemini-1.5-pro-latest': {'input': 1.25, 'output': 5.0},
}

DEFAULT_MODEL = 'gemini-2.5-pro'

import streamlit as st

def save_api_key(api_key):
    """Save API key to local file for persistence."""
    try:
        with open(API_KEY_FILE, 'w') as f:
            json.dump({'api_key': api_key}, f)
        return True
    except Exception as e:
        st.error(f"Error saving API key: {e}")
        return False

def load_api_key():
    """Load API key from local file."""
    try:
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, 'r') as f:
                data = json.load(f)
                return data.get('api_key', '')
        return ''
    except Exception as e:
        st.error(f"Error loading API key: {e}")
        return ''

def delete_api_key():
    """Delete stored API key file."""
    try:
        if os.path.exists(API_KEY_FILE):
            os.remove(API_KEY_FILE)
        return True
    except Exception as e:
        st.error(f"Error deleting API key: {e}")
        return False

def list_available_models():
    """Lists available current Gemini models that support content generation."""
    if 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
        return list(GEMINI_MODELS.keys())  # Return hardcoded list of current models
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.replace('models/', '')
                # Only include current models we have pricing for
                if model_name in GEMINI_MODELS:
                    available_models.append(model_name)
        # If no models found via API, return our hardcoded list
        return available_models if available_models else list(GEMINI_MODELS.keys())
    except Exception as e:
        st.error(f"Could not list models: {e}")
        return list(GEMINI_MODELS.keys())

def get_gemini_model(model_name: str, api_key: str = None):
    """Initializes and returns a configured Gemini model client.
    If api_key is provided, it uses that. Otherwise, it tries st.session_state (for Streamlit app).
    """
    if api_key is None:
        if 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
            st.error("Gemini API Key not set. Please enter your key in the sidebar.")
            st.stop()
        api_key = st.session_state.gemini_api_key
    
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8192,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction="You are a helpful assistant that helps legal researchers analyze legal texts."
    )
    return model
