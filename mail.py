#!/usr/bin/env python

import smtplib
file=open('accuracy.txt','r')


accuracy=file.read()
file.close()

Text="THE ACCURACY OF THE MODEL IS"

message=Text+" "+accuracy
s=smtplib.SMTP(host='smtp.gmail.com',port=587)

s.starttls()
s.login("forallchallenge@gmail.com","Ankush@12345")

s.sendmail("forallchallenge@gmail.com","forallchallenge@gmail.com",message)
s.quit()





