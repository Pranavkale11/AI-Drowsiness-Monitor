"""
sos/config.py — SOS Configuration
====================================
Loads Twilio credentials from environment variables (and optionally from a
.env file via python-dotenv).  Credentials are NEVER printed, logged, or
returned outside of private accessors — only their presence is exposed.

Environment variables consumed:
    TWILIO_ACCOUNT_SID      — Twilio Account SID (starts with "AC…")
    TWILIO_AUTH_TOKEN       — Twilio Auth Token (keep secret; never log)
    TWILIO_PHONE_NUMBER     — Your Twilio SMS-capable phone number (E.164)
    EMERGENCY_PHONE_NUMBER  — Fallback emergency recipient (E.164)
"""

import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def load_env(env_path: Optional[Path] = None) -> None:
    """Load a .env file using python-dotenv (if installed).

    Call this once at application startup.  If the file does not exist or
    python-dotenv is not installed, the function exits silently — the app
    will still work as long as the environment variables are set via other
    means (system env, Streamlit secrets, CI/CD injection, etc.).

    Args:
        env_path: Optional explicit path to the .env file.  Defaults to
                  ``<project_root>/.env`` (two levels above this module).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[SOS Config] python-dotenv is not installed; skipping .env load.")
        return

    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"[SOS Config] Loaded environment variables from {env_path}")
    else:
        print(
            f"[SOS Config] No .env file found at {env_path}. "
            "Using system environment variables."
        )


# ---------------------------------------------------------------------------
# Credential accessors (private values — only read once, never cached)
# ---------------------------------------------------------------------------

def get_twilio_account_sid() -> str:
    """Return the Twilio Account SID from the environment."""
    return os.getenv("TWILIO_ACCOUNT_SID", "").strip()


def get_twilio_auth_token() -> str:
    """Return the Twilio Auth Token from the environment.

    Warning:
        Never log, print, or display this value.
    """
    return os.getenv("TWILIO_AUTH_TOKEN", "").strip()


def get_twilio_phone_number() -> str:
    """Return the Twilio sender phone number from the environment."""
    return os.getenv("TWILIO_PHONE_NUMBER", "").strip()


def get_emergency_phone_number() -> str:
    """Return the fallback emergency recipient number from the environment."""
    return os.getenv("EMERGENCY_PHONE_NUMBER", "").strip()


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def is_sos_configured() -> bool:
    """Return True when all three required Twilio credentials are present."""
    return bool(
        get_twilio_account_sid()
        and get_twilio_auth_token()
        and get_twilio_phone_number()
    )


def get_config_status() -> dict:
    """Return a safe (no secret values) summary of the SOS configuration.

    Returns:
        A dict with boolean flags indicating which credentials are set.
        Auth token value is never included — only its presence is reported.
    """
    return {
        "account_sid_set": bool(get_twilio_account_sid()),
        "auth_token_set": bool(get_twilio_auth_token()),
        "phone_number_set": bool(get_twilio_phone_number()),
        "emergency_number_env_set": bool(get_emergency_phone_number()),
        "is_fully_configured": is_sos_configured(),
    }
