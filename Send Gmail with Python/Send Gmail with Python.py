############################send Gmail with Python#############################
# Will need to change Gmail lesssecureapps to low
# https://myaccount.google.com/lesssecureapps

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

gmail_user = 'account' # xxxx@gmail.com
gmail_password = 'password' 

msg = MIMEMultipart()
msg['Subject'] = 'Stock List'
msg['From'] = gmail_user
msg['To'] = '' # xxxx@gmail.com, xxxx@hotmail.com
text = "'Sent with Python'"
msg.attach(MIMEText(text))

filename = "xxx.csv" # file path
part = MIMEBase('application', "octet-stream")
part.set_payload(open(filename, "rb").read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', 'attachment; filename='+filename)
msg.attach(part)

server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
server.ehlo()
server.login(gmail_user, gmail_password)
server.send_message(msg)
server.quit()

