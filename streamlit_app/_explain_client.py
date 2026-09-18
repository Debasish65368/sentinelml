"""
streamlit_app/_explain_client.py — Pure helper functions for the /explain request.

This module contains only the logic needed to build and authenticate the
/explain API call. It has no Streamlit imports, so it can be imported and
tested in isolation without a running Streamlit session.

Imported by streamlit_app/app.py.
"""
import os

import requests


def get_sentinel_api_key(st_secrets=None):
    """Return the SENTINEL_API_KEY for /explain requests.

    Resolution order:
    1. st_secrets (Streamlit's st.secrets dict-like) — used on Streamlit Cloud.
    2. Environment variable SENTINEL_API_KEY — used locally via .env or shell export.

    Accepts st_secrets as a parameter (injected by app.py so this module does not
    import streamlit) to keep this module testable in isolation.

    Returns None if the key is not configured in either location.
    The key value is never logged, printed, or shown in the UI.
    """
    # Streamlit secrets (available on Streamlit Community Cloud)
    if st_secrets is not None:
        try:
            key = st_secrets.get("SENTINEL_API_KEY")
            if key:
                return key
        except Exception:
            pass
    # Fallback: environment variable (loaded by dotenv in app.py at startup)
    return os.getenv("SENTINEL_API_KEY") or None


def build_explain_headers(api_key):
    """Return the headers dict for an authenticated /explain POST request.

    Args:
        api_key: The SENTINEL_API_KEY string (must not be None).

    Returns:
        {"X-API-Key": api_key}
    """
    return {"X-API-Key": api_key}
