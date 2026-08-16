from unittest.mock import MagicMock

from orbit.config import settings
from orbit.email_agent.mailer import build_message, send_email


def test_build_message_without_attachment():
    message = build_message("friend@example.com", "Hello", "Just saying hi.")

    assert message["To"] == "friend@example.com"
    assert message["Subject"] == "Hello"
    assert message.get_content().strip() == "Just saying hi."
    assert list(message.iter_attachments()) == []


def test_build_message_with_attachment(tmp_path):
    attachment = tmp_path / "resume.txt"
    attachment.write_text("resume content")

    message = build_message("friend@example.com", "Resume", "See attached.", attachment)
    attachments = list(message.iter_attachments())

    assert len(attachments) == 1
    assert attachments[0].get_filename() == "resume.txt"
    assert attachments[0].get_content() == b"resume content"


def test_send_email_uses_configured_smtp_server(monkeypatch):
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_port", 587)
    monkeypatch.setattr(settings, "smtp_username", "me@example.com")
    monkeypatch.setattr(settings, "smtp_password", "secret")
    monkeypatch.setattr(settings, "smtp_use_tls", True)

    server = MagicMock()
    server.__enter__.return_value = server
    smtp_cls = MagicMock(return_value=server)
    monkeypatch.setattr("orbit.email_agent.mailer.smtplib.SMTP", smtp_cls)

    send_email("friend@example.com", "Hello", "Just saying hi.")

    smtp_cls.assert_called_once_with("smtp.example.com", 587)
    server.starttls.assert_called_once()
    server.login.assert_called_once_with("me@example.com", "secret")
    server.send_message.assert_called_once()
