
import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import json
import sqlite3
from sqlalchemy import create_engine
import tempfile

load_dotenv()

# Database configuration with environment-based selection
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL:
    # Production: Use PostgreSQL (Neon)
    DB_ENGINE = create_engine(DATABASE_URL)
    DB_TYPE = 'postgresql'
    DB_NAME = None  # Will use engine for connections
else:
    # Local development: Use SQLite
    DB_NAME = os.path.join(os.path.dirname(__file__), "parquet", "scc_cases.db")
    DB_ENGINE = None
    DB_TYPE = 'sqlite'

# API Key configuration
if DATABASE_URL:
    # Production: Use environment variable or Streamlit secrets
    API_KEY_FILE = None  # Will handle via environment/secrets
else:
    # Local development: Use local file
    API_KEY_FILE = os.path.join(os.path.dirname(__file__), ".api_key.json")

# Updated pricing for current Gemini models (2025)
GEMINI_MODELS = {
    'gemini-2.5-pro': {'input': 1.25, 'output': 10.0, 'input_high_volume': 2.5, 'output_high_volume': 15.0},
    'gemini-2.5-flash': {'input': 0.10, 'output': 0.40},
    'gemini-2.5-flash-lite': {'input': 0.075, 'output': 0.30},
    'gemini-1.5-pro-latest': {'input': 1.25, 'output': 5.0},
}

DEFAULT_MODEL = 'gemini-2.5-pro'

def get_database_connection():
    """Get database connection based on environment (SQLite or PostgreSQL)"""
    if DB_TYPE == 'postgresql':
        return DB_ENGINE.connect()
    else:
        return sqlite3.connect(DB_NAME)

def execute_sql(query, params=None, fetch=False):
    """Execute SQL query with proper connection handling"""
    if DB_TYPE == 'postgresql':
        with DB_ENGINE.connect() as conn:
            result = conn.execute(query, params or ())
            if fetch:
                return result.fetchall()
            conn.commit()
            return result
    else:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        if params:
            result = cursor.execute(query, params)
        else:
            result = cursor.execute(query)
        
        if fetch:
            data = result.fetchall()
            conn.close()
            return data
        else:
            conn.commit()
            conn.close()
            return result

import streamlit as st

def save_api_key(api_key):
    """Save API key with environment-aware storage."""
    if DB_TYPE == 'postgresql':
        # Production: Can't save to file, store in session only
        st.session_state.gemini_api_key = api_key
        st.info("API key saved for this session. Note: In production, consider using environment variables.")
        return True
    else:
        # Local development: Save to file
        try:
            with open(API_KEY_FILE, 'w') as f:
                json.dump({'api_key': api_key}, f)
            return True
        except Exception as e:
            st.error(f"Error saving API key: {e}")
            return False

def load_api_key():
    """Load API key from environment, Streamlit secrets, or local file."""
    # First, try environment variable (for production)
    env_key = os.getenv('GEMINI_API_KEY')
    if env_key:
        return env_key
    
    # Second, try Streamlit secrets (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except:
        pass
    
    # Third, try local file (for development)
    if API_KEY_FILE and os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, 'r') as f:
                data = json.load(f)
                return data.get('api_key', '')
        except Exception as e:
            if DB_TYPE == 'sqlite':  # Only show error in local development
                st.error(f"Error loading API key: {e}")
    
    return ''

def delete_api_key():
    """Delete stored API key."""
    if DB_TYPE == 'postgresql':
        # Production: Clear from session
        if 'gemini_api_key' in st.session_state:
            del st.session_state.gemini_api_key
        st.info("API key cleared from session.")
        return True
    else:
        # Local development: Delete file
        try:
            if API_KEY_FILE and os.path.exists(API_KEY_FILE):
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
