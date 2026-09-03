"""
sos/twilio_service.py — Twilio SMS Service
============================================
A clean, Streamlit-independent abstraction for sending SMS alerts via the
Twilio Programmable Messaging API.

Public API:
    send_sms(account_sid, auth_token, from_number, to_number, body)
        -> tuple[bool, str, dict]

    compose_auto_sos_message(timestamp, location=None)  -> str
    compose_manual_sos_message(timestamp, location=None) -> str
    compose_test_message(timestamp)                      -> str

The module never logs or surfaces the auth_token.  All exceptions from the
Twilio SDK are caught and returned as (False, friendly_message, error_dict).
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Message composers
# ---------------------------------------------------------------------------

def compose_auto_sos_message(
    timestamp: str,
    location: Optional[str] = None,
) -> str:
    """Build the automatic SOS alert message body.

    Args:
        timestamp: Human-readable time string (e.g. "2026-09-01 23:45:00").
        location:  Optional Google Maps URL or location description.

    Returns:
        Complete SMS body text.
    """
    lines = [
        "🚨 DRIVER EMERGENCY ALERT",
        "",
        "Severe driver drowsiness was detected by the AI Driver Monitoring System.",
        "The driver did not respond to warnings and a timed SOS countdown was triggered.",
        "",
        "Please check the driver immediately.",
        "",
        f"Time: {timestamp}",
    ]
    if location:
        lines.append(f"Location: {location}")
    lines += [
        "",
        "This is an automated emergency alert from DrowseGuard AI.",
    ]
    return "\n".join(lines)


def compose_manual_sos_message(
    timestamp: str,
    location: Optional[str] = None,
) -> str:
    """Build the manual SOS alert message body.

    Args:
        timestamp: Human-readable time string.
        location:  Optional Google Maps URL or location description.

    Returns:
        Complete SMS body text.
    """
    lines = [
        "🚨 MANUAL DRIVER EMERGENCY ALERT",
        "",
        "The driver manually triggered an SOS alert from the AI Driver Monitoring System.",
        "",
        "Please check the driver immediately.",
        "",
        f"Time: {timestamp}",
    ]
    if location:
        lines.append(f"Location: {location}")
    lines += [
        "",
        "This is a manual emergency alert from DrowseGuard AI.",
    ]
    return "\n".join(lines)


def compose_test_message(timestamp: str) -> str:
    """Build a non-emergency test SMS message body.

    Args:
        timestamp: Human-readable time string.

    Returns:
        Complete SMS body text.
    """
    return (
        "TEST MESSAGE — AI Driver Drowsiness Monitoring System\n\n"
        "This is a Twilio SMS integration test. No emergency has been detected.\n"
        "Your SOS emergency alert system is configured correctly.\n\n"
        f"Time: {timestamp}\n\n"
        "DrowseGuard AI"
    )


# ---------------------------------------------------------------------------
# SMS sender
# ---------------------------------------------------------------------------

def send_sms(
    account_sid: str,
    auth_token: str,
    from_number: str,
    to_number: str,
    body: str,
) -> tuple[bool, str, dict]:
    """Send an SMS message via the Twilio REST API.

    Args:
        account_sid:  Twilio Account SID.
        auth_token:   Twilio Auth Token (never logged).
        from_number:  Twilio phone number in E.164 format.
        to_number:    Recipient phone number in E.164 format.
        body:         SMS message body.

    Returns:
        A three-tuple of:
            success (bool)       — True if Twilio accepted the message.
            message (str)        — Human-readable result (no secrets).
            response_dict (dict) — Raw fields from Twilio response or
                                   error information.
    """
    # --- SDK availability check ---
    try:
        from twilio.rest import Client
        from twilio.base.exceptions import TwilioRestException
    except ImportError:
        return (
            False,
            "Twilio SDK is not installed. Run: pip install twilio",
            {"error": "ImportError"},
        )

    # --- Credential presence check ---
    if not account_sid or not auth_token or not from_number:
        missing = [
            name
            for name, val in [
                ("TWILIO_ACCOUNT_SID", account_sid),
                ("TWILIO_AUTH_TOKEN", auth_token),
                ("TWILIO_PHONE_NUMBER", from_number),
            ]
            if not val
        ]
        return (
            False,
            f"Missing Twilio credentials: {', '.join(missing)}. "
            "Check your .env file or environment variables.",
            {"missing_credentials": missing},
        )

    if not to_number:
        return (
            False,
            "No emergency phone number configured. "
            "Add a number in the sidebar and click 'Save'.",
            {"error": "missing_recipient"},
        )

    # --- Send via Twilio ---
    try:
        client = Client(account_sid, auth_token)
        msg = client.messages.create(body=body, from_=from_number, to=to_number)

        # Fetch for the most up-to-date status
        fetched = client.messages(msg.sid).fetch()
        response: dict = {
            "sid": fetched.sid,
            "status": fetched.status,
            "to": fetched.to,
            "from": fetched.from_,
            "error_code": fetched.error_code,
            "error_message": fetched.error_message,
            "date_created": str(fetched.date_created),
            "date_sent": str(fetched.date_sent),
        }
        return True, "Emergency SMS sent successfully.", response

    except Exception as exc:  # noqa: BLE001
        error_code: Optional[int] = getattr(exc, "code", None)
        # Redact auth token from any error string before logging
        raw_error = str(exc)
        if auth_token and auth_token in raw_error:
            raw_error = raw_error.replace(auth_token, "[REDACTED]")

        friendly = _get_friendly_error(error_code, raw_error, to_number)
        return (
            False,
            friendly,
            {
                "error_code": error_code,
                "error_message": raw_error,
                "more_info": getattr(exc, "more_info", None),
            },
        )


# ---------------------------------------------------------------------------
# Private: friendly error messages
# ---------------------------------------------------------------------------

def _get_friendly_error(
    error_code: Optional[int],
    error_message: str,
    to_number: str,
) -> str:
    """Return a beginner-friendly explanation for common Twilio errors."""
    msg_lower = (error_message or "").lower()

    if error_code == 20003 or "authenticate" in msg_lower or "auth" in msg_lower:
        return (
            "Twilio rejected your credentials. "
            "Check TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in your .env file."
        )
    if error_code == 21608:
        return (
            "Trial accounts can only send SMS to verified recipient numbers. "
            "Verify the destination number in the Twilio Console "
            "(Phone Numbers → Verified Caller IDs)."
        )
    if error_code == 21606:
        return (
            "The Twilio sender number is invalid or not SMS-capable for this account. "
            "Check TWILIO_PHONE_NUMBER in your .env file."
        )
    if error_code == 21614:
        return "The destination number is not a valid mobile number for SMS delivery."
    if error_code == 30007:
        return "The carrier likely filtered or blocked the SMS. Try a different message format."
    if error_code == 30008:
        return "The destination handset may be unreachable, roaming, or temporarily unavailable."
    if error_code == 30034:
        return (
            "The carrier blocked the message because the sender is not properly registered "
            "for that route. Check Twilio Console messaging logs."
        )
    if to_number.startswith("+91"):
        return (
            "India delivery can be restricted by carrier rules. Trial accounts add extra limits. "
            "Verify the recipient number in the Twilio Console and check messaging logs."
        )
    return (
        f"Delivery failed (Twilio error code: {error_code}). "
        "Check Twilio Console → Monitor → logs for details."
    )


# ---------------------------------------------------------------------------
# Voice Call Services
# ---------------------------------------------------------------------------

def compose_auto_voice_twiml(timestamp: str, location: Optional[str] = None) -> str:
    """Build the TwiML for an automatic SOS voice call."""
    loc_phrase = "Location information is unavailable."
    if location:
        loc_phrase = "Check the accompanying SMS for the Google Maps location link."
        
    return (
        f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        f"<Response>\n"
        f"    <Say voice=\"alice\" language=\"en-US\">"
        f"        Emergency Alert from Drowse Guard A I. "
        f"        Severe driver drowsiness has been detected, and the driver did not respond to warnings. "
        f"        Please check on the driver immediately. "
        f"        {loc_phrase} "
        f"        This alert was generated at {timestamp}. "
        f"        Repeating. Emergency Alert from Drowse Guard A I. "
        f"        Severe driver drowsiness has been detected. "
        f"    </Say>\n"
        f"</Response>"
    )


def compose_manual_voice_twiml(timestamp: str, location: Optional[str] = None) -> str:
    """Build the TwiML for a manual SOS voice call."""
    loc_phrase = "Location information is unavailable."
    if location:
        loc_phrase = "Check the accompanying SMS for the Google Maps location link."
        
    return (
        f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        f"<Response>\n"
        f"    <Say voice=\"alice\" language=\"en-US\">"
        f"        Manual Emergency Alert from Drowse Guard A I. "
        f"        The driver has manually triggered an S O S alert. "
        f"        Please check on the driver immediately. "
        f"        {loc_phrase} "
        f"        This alert was generated at {timestamp}. "
        f"        Repeating. Manual Emergency Alert from Drowse Guard A I. "
        f"        The driver has manually triggered an S O S alert. "
        f"    </Say>\n"
        f"</Response>"
    )


def compose_test_voice_twiml(timestamp: str) -> str:
    """Build the TwiML for a test voice call."""
    return (
        f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        f"<Response>\n"
        f"    <Say voice=\"alice\" language=\"en-US\">"
        f"        This is a test message from the Drowse Guard A I monitoring system. "
        f"        No emergency has been detected. Your S O S voice alert system is configured correctly. "
        f"        Goodbye. "
        f"    </Say>\n"
        f"</Response>"
    )


def make_voice_call(
    account_sid: str,
    auth_token: str,
    from_number: str,
    to_number: str,
    twiml: str,
) -> tuple[bool, str, dict]:
    """Make a programmable voice call via the Twilio REST API.

    Args:
        account_sid:  Twilio Account SID.
        auth_token:   Twilio Auth Token (never logged).
        from_number:  Twilio phone number in E.164 format.
        to_number:    Recipient phone number in E.164 format.
        twiml:        TwiML string to execute on the call.

    Returns:
        A three-tuple of:
            success (bool)       — True if Twilio accepted the call.
            message (str)        — Human-readable result (no secrets).
            response_dict (dict) — Raw fields from Twilio response or error info.
    """
    try:
        from twilio.rest import Client
        from twilio.base.exceptions import TwilioRestException
    except ImportError:
        return (
            False,
            "Twilio SDK is not installed. Run: pip install twilio",
            {"error": "ImportError"},
        )

    if not account_sid or not auth_token or not from_number:
        missing = [
            name
            for name, val in [
                ("TWILIO_ACCOUNT_SID", account_sid),
                ("TWILIO_AUTH_TOKEN", auth_token),
                ("TWILIO_PHONE_NUMBER", from_number),
            ]
            if not val
        ]
        return (
            False,
            f"Missing Twilio credentials: {', '.join(missing)}. ",
            {"missing_credentials": missing},
        )

    if not to_number:
        return (
            False,
            "No emergency phone number configured. ",
            {"error": "missing_recipient"},
        )

    try:
        client = Client(account_sid, auth_token)
        call = client.calls.create(
            twiml=twiml,
            to=to_number,
            from_=from_number
        )

        response: dict = {
            "sid": call.sid,
            "status": call.status,
            "to": call.to,
            "from": call.from_,
            "date_created": str(call.date_created),
        }
        return True, "Emergency Voice Call initiated successfully.", response

    except Exception as exc:  # noqa: BLE001
        error_code: Optional[int] = getattr(exc, "code", None)
        raw_error = str(exc)
        if auth_token and auth_token in raw_error:
            raw_error = raw_error.replace(auth_token, "[REDACTED]")

        friendly = _get_friendly_error(error_code, raw_error, to_number)
        return (
            False,
            friendly,
            {
                "error_code": error_code,
                "error_message": raw_error,
                "more_info": getattr(exc, "more_info", None),
            },
        )
