from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.util import ngrams
from collections import Counter
import pandas as pd
from nltk import everygrams
import re
import numpy as np
import random
import math
import operator
import string
import nltk

file1 = open('uppertolower.txt', 'r',encoding='UTF8')
Lines = file1.read()


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


Lines=remove_emoji(Lines)
b=Lines
b=b.replace('\n','')
tnew = sent_tokenize(b)


for i in range (len(tnew)):
  tnew[i] = re.sub(r'[^A-Za-z\s\']+', "", tnew[i])
  tnew[i] = '<s> '+tnew[i]+' </s>'
train = tnew[:10000] # 80% of dataset is train
test = tnew[10000:10000+2500] # 20% of dataset is testprint(len(tnew))

## Train dataset
# fdist0 = {} #unigram
# for i in range (len(train)):
#   tokens = train[i].split()
#   unigrams = ngrams(tokens,1)
#   fdist = dict(nltk.FreqDist(unigrams))
#   fdist0 = dict(Counter(fdist)+Counter(fdist0))


fdist1 = {} #bigram
for i in range (len(train)):
  tokens = train[i].split()
  bigrams = nltk.bigrams(tokens)
  fdist = dict(nltk.FreqDist(bigrams))
  fdist1 = dict(Counter(fdist)+Counter(fdist1))


fdist2 = {} #trigram
for i in range (len(train)):
  tokens = train[i].split()
  trigrams = nltk.trigrams(tokens)
  fdist = dict(nltk.FreqDist(trigrams))
  fdist2 = dict(Counter(fdist)+Counter(fdist2))

# from nltk import everygrams
# fdist3 = {} #quadgram
# for i in range (len(train)):
#   tokens = train[i].split()
#   fourgrams = list(everygrams(tokens,4,4)) 
#   fdist = dict(nltk.FreqDist(fourgrams))
#   fdist3 = dict(Counter(fdist)+Counter(fdist3))

# print("The total number of unigrams in train set are: ", len(fdist0))
# print("The total number of bigrams in train set are: ", len(fdist1))
# print("The total number of trigrams in train set are: ", len(fdist2))
# print("The total number of quadgrams in train set are: ", len(fdist3))


## Test dataset
# fdist0_t = {} #unigram
# for i in range (len(test)):
#   tokens = test[i].split()
#   unigrams = ngrams(tokens,1)
#   fdist = dict(nltk.FreqDist(unigrams))
#   fdist0_t = dict(Counter(fdist)+Counter(fdist0_t))


# fdist1_t = {} #bigram
# for i in range (len(test)):
#   tokens = test[i].split()
#   bigrams = nltk.ngrams(tokens,2)
#   fdist = dict(nltk.FreqDist(bigrams))
#   fdist1_t = dict(Counter(fdist)+Counter(fdist1_t))


fdist2_t = {} #trigram
for i in range (len(test)):
  tokens = test[i].split()
  trigrams = nltk.trigrams(tokens)
  fdist = dict(nltk.FreqDist(trigrams))
  fdist2_t = dict(Counter(fdist)+Counter(fdist2_t))


# fdist3_t = {} #quadgram
# for i in range (len(test)):
#   tokens = test[i].split()
#   fourgrams = list(everygrams(tokens,4,4)) 
#   fdist = dict(nltk.FreqDist(fourgrams))
#   fdist3_t = dict(Counter(fdist)+Counter(fdist3_t))

# key,val = fdist0_t.keys(),fdist0_t.values()
# dd = {"word":key,"count":val}
# df = pd.DataFrame.from_dict(dd)
# df.to_csv("unigram_new.csv")

'''
fdist0 = pd.read_csv("unigram_new.csv")
kgram = fdist0['kgram']
#print(kgram[1])
count = fdist0['count']
temp = []
for i in range(len(kgram)):
    temp.append((kgram[i],int(count[i])))
#print(temp)
fdist0 = dict(temp)

fdist1 = pd.read_csv("bigram_test.csv")
kgram = fdist1['kgram']
#print(kgram[1])
count = fdist1['count']
temp = []
for i in range(len(kgram)):
    temp.append((kgram[i],int(count[i])))
#print(temp)
fdist1 = dict(temp)

fdist2 = pd.read_csv("trigram_test.csv")
kgram = fdist2['kgram']
#print(kgram[1])
count = fdist2['count']
temp = []
for i in range(len(kgram)):
    temp.append((kgram[i],int(count[i])))
#print(temp)
fdist2 = dict(temp)

fdist3 = pd.read_csv("quadgram_test.csv")
kgram = fdist3['kgram']
#print(kgram[1])
count = fdist3['count']
temp = []
for i in range(len(kgram)):
    temp.append((kgram[i],int(count[i])))
#print(temp)
fdist3 = dict(temp)
'''


#### MLEs
trainstr = re.sub(r'[^A-Za-z\s\']+', "", b)
ls = trainstr.split()
dictfinal = {}
for i in range (len(ls)):
  if ls[i] not in dictfinal:
    dictfinal[ls[i]] = 1
  else:
    dictfinal[ls[i]] += 1
Vocab = len(dictfinal)
Token = sum(dictfinal.values())
#Vocab


d=dictfinal
mle_uni = []
mle_unis = []
mle_bi = []
mle_tri = []
mle_quad = []

def MLEunigram(w1):
  if w1 not in d:
    mle_uni.append(1/(Vocab+Token))
    return 1/(Vocab+Token) # Add 1 Smoothing
  else:
    mle_uni.append((d[w1]+1)/(Vocab+Token))
    return (d[w1]+1)/(Vocab+Token) # Add 1 Smoothing


def MLEunigram_nosmooth(w1):
  if w1 not in d:
    mle_unis.append(0)
    return 0 # Add 1 Smoothing
  else:
    mle_unis.append((d[w1])/(Token))
    return (d[w1])/(Token) # Add 1 Smoothing


def MLEbigram(w1,w2):
  if (w1+" "+w2) not in fdist1:
    if w1 in d.keys():
      return 1/(d[w1]+Vocab) # Add 1 Smoothing
    else:
      return 1/(Vocab) # Add 1 Smoothing
  else:
    if w1 in d.keys():
      return (fdist1(w1+" "+w2)+1)/(d[w1]+Vocab) # Add 1 Smoothing
    else:
      return (fdist1(w1+" "+w2)+1)/(Vocab) # Add 1 Smoothing


def MLEbigram_nosmooth(w1,w2):
  if (w1+" "+w2) not in fdist1:
    if w1 in d.keys():
      return 0 # Add 1 Smoothing
    else:
      return 0 # Add 1 Smoothing
  else:
    if w1 in d.keys():
      return (fdist1(w1+" "+w2))/(d[w1]) # Add 1 Smoothing
    # else:
    #   return (fdist1(w1+" "+w2))/(Vocab) # Add 1 Smoothing



def MLEtrigram(w1,w2,w3):
  if (w1+" "+w2+" "+w3) not in fdist2:
    if w1+" "+w2 in fdist1.keys():
      return 1/(fdist1[w1+" "+w2]+Vocab) # Add 1 Smoothing
    else:
      return 1/(Vocab) # Add 1 Smoothing
  else:
    if w1+" "+w2 in fdist1.keys():
      return (fdist2(w1+" "+w2+" "+w3)+1)/(fdist1[w1+" "+w2]+Vocab) # Add 1 Smoothing
    else:
      return (fdist2(w1+" "+w2+" "+w3)+1)/(Vocab) # Add 1 Smoothing


def MLEtrigram_nosmooth(w1,w2,w3):
  if (w1+" "+w2+" "+w3) not in fdist2:
    if w1+" "+w2 in fdist1.keys():
      return 0# Add 1 Smoothing
    else:
      return 0 # Add 1 Smoothing
  else:
    if w1+" "+w2 in fdist1.keys():
      return (fdist2(w1+" "+w2+" "+w3))/(fdist1[w1+" "+w2]) # Add 1 Smoothing
    else:
      return (fdist2(w1+" "+w2+" "+w3))/(Vocab) # Add 1 Smoothing



def MLEquadgram(w1,w2,w3,w4):
  if (w1+" "+w2+" "+w3+" "+w4) not in fdist3:
    if w1+" "+w2 + " "+w3 in fdist2.keys():
      return 1/(fdist2[w1+" "+w2+" "+w3]+Vocab) # Add 1 Smoothing
    else:
      return 1/(Vocab) # Add 1 Smoothing
  else:
    if w1+" "+w2 + " "+w3 in fdist2.keys():
      return (fdist3(w1+" "+w2+" "+w3+" "+w4)+1)/(fdist2[w1+" "+w2+" "+w3]+Vocab) # Add 1 Smoothing
    else:
      return (fdist3(w1+" "+w2+" "+w3+" "+w4)+1)/(Vocab) # Add 1 Smoothing



def MLEquadgram_nosmooth(w1,w2,w3,w4):
  if (w1+" "+w2+" "+w3+" "+w4) not in fdist3:
    if w1+" "+w2 + " "+w3 in fdist2.keys():
      return 0 # Add 1 Smoothing
    else:
      return 0 # Add 1 Smoothing
  else:
    if w1+" "+w2 + " "+w3 in fdist2.keys():
      return (fdist3(w1+" "+w2+" "+w3+" "+w4)+1)/(fdist2[w1+" "+w2+" "+w3]+Vocab) # Add 1 Smoothing
    # else:
    #   return (fdist3(w1+" "+w2+" "+w3+" "+w4)+1)/(Vocab) # Add 1 Smoothing


#### Perplexity
def unigramsentenceprob(sentence):
  sentence_probability_log_sum = 0
  for word in sentence:
    x = MLEunigram_nosmooth(word)
    sentence_probability_log_sum += math.log(x,2)
  return math.pow(2, sentence_probability_log_sum)


def bigramsentenceprob(sentence):
  bigram_sentence_probability_log_sum = 0
  previous_word = None
  for word in sentence:
    if previous_word!=None:
      x = MLEbigram(previous_word,word)
      bigram_sentence_probability_log_sum += math.log(x,2)
    previous_word = word
  return math.pow(2, bigram_sentence_probability_log_sum)


def trigramsentenceprob(sentence):
  trigram_sentence_probability_log_sum = 0
  previous_word = None
  previous_previous_word = None
  for word in sentence:
    if previous_word!=None and previous_previous_word!=None:
      x = MLEtrigram(previous_previous_word,previous_word,word)
      trigram_sentence_probability_log_sum += math.log(x,2)
    previous_previous_word = previous_word
    previous_word = word
  return math.pow(2, trigram_sentence_probability_log_sum)

def quadgramsentenceprob(sentence):
  quadgram_sentence_probability_log_sum = 0
  previous_word = None
  previous_previous_word = None
  previous_previous_previous_word = None
  for word in sentence:
    if previous_word!=None and previous_previous_word!=None and previous_previous_previous_word!=None :
      x = MLEquadgram(previous_previous_previous_word,previous_previous_word,previous_word,word)
      quadgram_sentence_probability_log_sum += math.log(x,2)
    previous_previous_previous_word = previous_previous_word
    previous_previous_word = previous_word
    previous_word = word
  return math.pow(2, quadgram_sentence_probability_log_sum)


# ### Calculate number of unigrams, etc....
def calculate_number_of_unigrams(sentences):
  unigram_count = 0
  for sentence in sentences:
    # remove two for <s> and </s>
    unigram_count += len(sentence) - 2
  return unigram_count

def calculate_number_of_bigrams(sentences):
  bigram_count = 0
  for sentence in sentences:
    bigram_count += len(sentence) - 1
  return bigram_count

def calculate_number_of_trigrams(sentences):
  trigram_count = 0
  for sentence in sentences:
    trigram_count += len(sentence) - 2
  return trigram_count

def calculate_number_of_quadgrams(sentences):
  quadgram_count = 0
  for sentence in sentences:
    quadgram_count += len(sentence) - 3
  return quadgram_count



# ## Call the above functions by passing train and test lists


def calculate_unigram_perplexity(sentences):
  unigram_count = calculate_number_of_unigrams(sentences)
  sentence_probability_log_sum = 0
  for sentence in sentences:
    try:
      sentence_probability_log_sum -= math.log(unigramsentenceprob(sentence), 2)
    except:
      sentence_probability_log_sum -= 0
  return math.pow(2, sentence_probability_log_sum / unigram_count)

def calculate_bigram_perplexity(sentences):
  bigram_count = calculate_number_of_bigrams(sentences)
  sentence_probability_log_sum = 0
  for sentence in sentences:
    try:
      sentence_probability_log_sum -= math.log(bigramsentenceprob(sentence), 2)
    except:
      sentence_probability_log_sum -= 0
  return math.pow(2, sentence_probability_log_sum / bigram_count)

def calculate_trigram_perplexity(sentences):
  trigram_count = calculate_number_of_trigrams(sentences)
  sentence_probability_log_sum = 0
  for sentence in sentences:
    try:
      sentence_probability_log_sum -= math.log(trigramsentenceprob(sentences), 2)
    except:
      sentence_probability_log_sum -= 0
  return math.pow(2, sentence_probability_log_sum / trigram_count)


def calculate_quadgram_perplexity(sentences):
  quadgram_count = calculate_number_of_quadgrams(sentences)
  sentence_probability_log_sum = 0
  for sentence in sentences:
    try:
      sentence_probability_log_sum -= math.log(quadgramsentenceprob(sentence), 2)
    except:
      sentence_probability_log_sum -= 0
  return math.pow(2, sentence_probability_log_sum / quadgram_count)



# sum, avg, count = 0, 0, 0
# for word in train:
#     x = MLEunigram_nosmooth(word)
#     count+=1
#     sum += x
# avgmle = sum/count
# print("MLE of test corpus with respect to unigram model is:", end = " ")
# print(avgmle)

# sum, avg, count = 0, 0, 0
# previous_word = None
# for word in train:
#   if previous_word!=None:
#     x = MLEbigram(previous_word,word)
#     count+=1
#     sum += x
#   previous_word = word
# avgmle = sum/count
# print("MLE of test corpus with respect to bigram model is:", end = " ")
# print(avgmle)

sum, avg, count = 0, 0, 0
previous_word = None
previous_previous_word = None
for word in train:
  if previous_word!=None and previous_previous_word!=None:
    x = MLEtrigram(previous_previous_word,previous_word,word)
    count+=1
    sum += x
  previous_previous_word = previous_word
  previous_word = word
avgmle = sum/count
print("MLE of test corpus with respect to trigram model is:", end = " ")
print(avgmle)

# sum, avg, count = 0, 0, 0
# previous_word = None
# previous_previous_word = None
# previous_previous_previous_word = None
# for word in train:
#     if previous_word!=None and previous_previous_word!=None and previous_previous_previous_word!=None :
#       x = MLEquadgram(previous_previous_previous_word,previous_previous_word,previous_word,word)
#       count+=1
#       sum += x
# previous_previous_previous_word = previous_previous_word
# previous_previous_word = previous_word
# previous_word = word
# avgmle = sum/count
# print("MLE of test corpus with respect to quadgram model is:", end = " ")
# print(avgmle)

# print("Perplexity of test corpus with respect to unigram model is:",end = " ")
# print(calculate_unigram_perplexity(test))
# print("Perplexity of test corpus with respect to bigram model is:",end = " ")
# print(calculate_bigram_perplexity(test))
print("Perplexity of test corpus with respect to trigram model is:",end = " ")
print(calculate_trigram_perplexity(test))
# print("Perplexity of test corpus with respect to quadgram model is:",end = " ")
# print(calculate_quadgram_perplexity(test))