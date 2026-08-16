import smtplib
from email.message import EmailMessage
from pathlib import Path

from orbit.config import settings


def build_message(to: str, subject: str, body: str, attachment: Path | None = None) -> EmailMessage:
    """Assemble an EmailMessage, attaching `attachment` as a generic binary
    file if given. Kept separate from send_email so tests can inspect the
    constructed message without touching a real SMTP connection."""
    message = EmailMessage()
    message["From"] = settings.smtp_username
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    if attachment is not None:
        data = attachment.read_bytes()
        message.add_attachment(
            data, maintype="application", subtype="octet-stream", filename=attachment.name
        )

    return message


def send_email(to: str, subject: str, body: str, attachment: Path | None = None) -> None:
    """Send an email via the configured SMTP server. Connects fresh per call --
    Email Agent sends are infrequent and user-confirmed, so connection pooling
    isn't worth the complexity."""
    message = build_message(to, subject, body, attachment)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        if settings.smtp_use_tls:
            server.starttls()
        server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(message)
