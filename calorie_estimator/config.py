import os
from dotenv import load_dotenv
import streamlit as st

def load_api_keys():
    """
    Load API keys from .env file or environment variables.
    
    Returns:
        dict: Dictionary containing API keys
    """
    # Try to load from .env file
    load_dotenv()
    
    # Get API key with fallback
    api_keys = {
        "nutrition_api_key": os.getenv("NUTRITION_API_KEY", "")
    }
    
    return api_keys

def get_nutrition_api_key():
    """
    Get the Calorie Ninja API key from environment or session state.
    
    Returns:
        str: Calorie Ninja API key
    """
    # Check if we already loaded the key in this session
    if "nutrition_api_key" in st.session_state and st.session_state.nutrition_api_key:
        return st.session_state.nutrition_api_key
    
    # Load from environment
    api_keys = load_api_keys()
    nutrition_api_key = api_keys["nutrition_api_key"]
    
    # Store in session state for reuse
    st.session_state.nutrition_api_key = nutrition_api_key
    
    return nutrition_api_key 