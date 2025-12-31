import streamlit as st

def setup_page(title: str, icon: str = "🎬"):
    """
    Common page setup including title, layout, and custom CSS.
    """
    if "page_setup_done" not in st.session_state:
        st.set_page_config(page_title=title, page_icon=icon, layout="wide")
        st.session_state.page_setup_done = True

    # Custom CSS for better styling (supports both light and dark mode)
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .card {
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            transition: transform 0.2s;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h3 {
            margin-top: 0;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            font-weight: 600;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header(title: str, description: str = None):
    """
    Renders a standardized header.
    """
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<p style="font-size: 1.1rem; opacity: 0.7; margin-bottom: 2rem;">{description}</p>', unsafe_allow_html=True)

def render_card(title: str, description: str, icon: str, button_text: str, key: str):
    """
    Renders a clickable card component.
    """
    st.markdown(
        f"""
        <div class="card">
            <h3>{icon} {title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button(button_text, key=key):
        return True
    return False

def render_api_key_status(provider: str, is_valid: bool):
    """
    Renders a small status indicator for API keys.
    """
    if is_valid:
        st.sidebar.markdown(f"✅ **{provider}**: Ready")
    else:
        st.sidebar.markdown(f"⚠️ **{provider}**: Not Set")

def get_api_key(provider: str):
    """
    Helper to get API key from session state securely.
    """
    key_map = {
        "OpenAI": "openai_api_key",
        "Claude": "claude_api_key",
        "Anthropic": "claude_api_key" 
    }
    key_name = key_map.get(provider)
    if key_name:
        return st.session_state.get(key_name, "")
    return ""

def validate_api_inputs(providers: list):
    """
    Checks if required API keys are present and shows a warning if not.
    Returns True if all required keys are present.
    """
    missing = []
    for provider in providers:
        key = get_api_key(provider)
        if not key:
            missing.append(provider)
    
    if missing:
        st.warning(f"⚠️ Please enter your API Key for: {', '.join(missing)} in the sidebar settings.")
        return False
    return True
