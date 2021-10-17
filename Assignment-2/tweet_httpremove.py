import csv
import pandas as pd
import re
from random import sample

filename_in = "Pollution_Devanshu.csv" # Input file
data = pd.read_csv(filename_in)


tweets = list(data["tweet"])
lang = list(data["language"])

lisco=[]

for i in range(len(tweets)):
    lisco.append([tweets[i],lang[i]])

# lisco1=sample(lisco,140) # Enter number of samples you want
lisco1=lisco

tweets_=[]
lang_=[]

for i in range(len(lisco1)):
    tweets_.append(lisco1[i][0])
    lang_.append(lisco1[i][1])

output_file="Pollution_http_remove.csv" # Output file
fields=['tweet','language']

twilang=[]

for i in range(len(tweets_)):
    tweet = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '', tweets_[i], flags=re.MULTILINE) # to remove links that start with HTTP/HTTPS in the tweet
    tweet = re.sub(r'http\S+', '', tweet,flags=re.MULTILINE)
    tweet = re.sub(r'[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '', tweet, flags=re.MULTILINE) # to remove other url links

    #tweet = ''.join(re.sub("(@[A-Za-z0–9]+)|([⁰-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split()) # for emojis

    tweet = re.sub(r"#(\w+)", ' ', tweet, flags=re.MULTILINE)
    tweet = re.sub(r"@(\w+)", ' ', tweet, flags=re.MULTILINE)

    twilang.append([tweet,lang_[i]])


with open(output_file,'w',encoding='UTF8',newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(twilang)




# clean_tweet = re.sub("@[A-Za-z0-9_]+","", tweets[0])
# clean_tweet = re.sub("#[A-Za-z0-9_]+","", clean_tweet)

# clean_tweet = re.sub(r"#(\w+)", ' ', tweets[0], flags=re.MULTILINE)
# clean_tweet = re.sub(r"@(\w+)", ' ', clean_tweet, flags=re.MULTILINE)