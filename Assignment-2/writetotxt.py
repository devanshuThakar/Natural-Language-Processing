import csv
import pandas as pd

filename_in = "Pollution_http_remove.csv" # Input file
data = pd.read_csv(filename_in)


tweets = list(data["tweet"])

file1 = open("Pollution.txt","w",encoding='UTF8')
for i in range(len(tweets)):
    file1.write(tweets[i]+"\n")