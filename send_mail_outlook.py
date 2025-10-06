# send_mail_outlook.py — Envío por Outlook/Office365 via SMTP
import os
import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path

def send_outlook_email(subject, body_html, to_addrs, attachment_path=None):
    smtp_server = os.environ.get("OUTLOOK_SMTP_SERVER", "smtp.office365.com")
    smtp_port   = int(os.environ.get("OUTLOOK_SMTP_PORT", "587"))
    user        = os.environ["OUTLOOK_USER"]
    password    = os.environ["OUTLOOK_PASS"]

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
    subject = "Prueba Reporte Trimestral"
    body = "<h1>Prueba</h1><p>Envio Outlook SMTP OK.</p>"
    to = os.environ.get("OUTLOOK_TO", "")
    if to:
        send_outlook_email(subject, body, to, None)
