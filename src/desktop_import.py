import os
import pandas as pd

desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

my_file = desktop + "/sms+spam+collection/SMSSpamCollection.txt" 

if os.path.isfile(my_file):
    with open(my_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        if line[:3] =="ham":
            data.append((0,line[4:]))
        else:
            data.append((1,line[5:]))
    data = pd.DataFrame(data, columns=["Label", "Text"])
    data.to_csv("SMS_HAM_SPAM.csv", index=False)
else:
    raise FileNotFoundError("path doesn't work or file doesn't exist")


    

    