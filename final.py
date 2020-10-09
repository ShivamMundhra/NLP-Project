import urllib.request, urllib.parse, urllib.error
import ssl
import json
import nltk
import re
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

# Extracting First text
site1 = 'http://www.gutenberg.org/files/63365/63365-0.txt'
print('Extracting ',site1)
uh1 = urllib.request.urlopen(site1)
data1 = uh1.read().decode('utf8')

# Extracting second text
site2 = 'http://www.gutenberg.org/files/63369/63369-0.txt'
print('Extracting ',site2)
uh2 = urllib.request.urlopen(site2)
data2 = uh2.read().decode('utf8')

# pre-processing
data1 = re.sub('^Section [1-9].', '', data1)
data1 = re.sub(r'==.*?==+', '', data1)
data1 = re.sub('[\(\[].*?[\)\]]', '', data1)
data1 = re.sub(r'[^a-zA-Z0-9\s]', '', data1)
data1 = data1.replace('\n', '')

data2 = re.sub('^Section [1-9].', '', data2)
data2 = re.sub(r'==.*?==+', '', data2)
data2 = re.sub('[\(\[].*?[\)\]]', '', data2)
data2 = re.sub(r'[^a-zA-Z0-9\s]', '', data2)
data2 = data2.replace('\n', '')

# tokenizing
token1 = nltk.word_tokenize(data1)
token2 = nltk.word_tokenize(data2)

# frequency Distribution
fdist1 = FreqDist(token1)
fdist2 = FreqDist(token2)

fig = plt.figure(figsize=(40, 30))
# plotting freq. Dist. of 20 most common words
fdist1.plot(20)
fdist2.plot(20)

# Creating Word Cloud without StopWords
wordcloud1 = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', stopwords = [], collocations=False).generate(data1)

plt.figure(figsize=(40, 30))
plt.imshow(wordcloud1, interpolation = 'bilinear')
# No axis details
plt.axis("off");

wordcloud2 = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', stopwords = [], collocations=False).generate(data2)

plt.figure(figsize=(40, 30))
plt.imshow(wordcloud2)
# No axis details
plt.axis("off");

# Finding Stopwords
stop_words = set(stopwords.words('english'))

# stopwords in data 1
sp1 = []
for w in token1:
    if w in stop_words:
        sp1.append(w)

# stopwords in data 2
sp2 = []
for w in token2:
    if w in stop_words:
        sp2.append(w)

# Creating Word Cloud Again with Stop words
modified_wordcloud1 = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', stopwords = sp1, collocations = False).generate(data1)
plt.figure(figsize=(40, 30))
plt.imshow(modified_wordcloud1)
# No axis details
plt.axis("off");


modified_wordcloud2 = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', stopwords = sp2, collocations = False).generate(data2)
plt.figure(figsize=(40, 30))
plt.imshow(modified_wordcloud2)
# No axis details
plt.axis("off");

#POS Tagging
word_list1 = [w for w in token1 if not w in sp1]
word_list2 = [w for w in token2 if not w in sp2]

tagged1 = nltk.pos_tag(word_list1)
tagged2 = nltk.pos_tag(word_list2)
