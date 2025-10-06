# send_mail_gmail.py — Envío por Gmail con App Password
import os
import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path

def send_gmail(subject, body_html, to_addrs, attachment_path=None):
    user = os.environ["GMAIL_USER"]
    password = os.environ["GMAIL_PASS"]
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addrs if isinstance(to_addrs, str) else ", ".join(to_addrs)
    msg["Date"] = formatdate(localtime=True)
    msg["Subject"] = subject
    msg.set_content("Este mensaje contiene contenido HTML.")
    msg.add_alternative(body_html, subtype="html")

    if attachment_path and Path(attachment_path).is_file():
        with open(attachment_path, "rb") as f:
            data = f.read()
        filename = Path(attachment_path).name
        msg.add_attachment(data, maintype="text", subtype="html", filename=filename)

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
    print("Correo enviado a:", msg["To"])

if __name__ == "__main__":
    subject = "Prueba Gmail"
    body = "<h1>Prueba desde Gmail</h1><p>Correo mandado por Python.</p>"
    to = os.environ.get("GMAIL_TO", "")
    if to:
        send_gmail(subject, body, to, None)
