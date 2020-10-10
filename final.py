import urllib.request, urllib.parse, urllib.error
import ssl
import json
import nltk
import re
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

def plot(X,Y,xlabel,ylabel,title):
	plt.bar(X, Y, tick_label = X, width = 0.8, color = ['red', 'green'])
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

def plotRelationShip(fd):

	# fd is the frequency distribution of each of word
    # word length is a list that will
	word_lengths = {}
    
    for i in fd.keys():
		if len(i) not in word_lengths.keys():
			word_lengths[len(i)] = fd[i]	#adding a new word length and its frequency
		else:
			word_lengths[len(i)] += fd[i]	#adding the frequency of already existing word length

	#X will contain all lengths of the word
	#Y will contain the corresponding frequency
	X = []
	Y = []

	for i in word_lengths.keys():
		X.append(i)

	X.sort()

	for i in X:
		Y.append(word_lengths[i])

	#Plotting a bar graph for recorded data
	xlabel = 'word length'
	ylabel = 'frequency'
	title = 'Relationship between word length and frequency'
	plot(X,Y,xlabel,ylabel,title)


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
data1 = data1.lower()
data1 = re.sub('^Section [1-9].', '', data1)
data1 = re.sub(r'==.*?==+', '', data1)
data1 = re.sub(r'CHAPTER \d+', '', data1)
data1 = re.sub('[\(\[].*?[\)\]]', '', data1)
data1 = re.sub(r'[^a-zA-Z0-9\s]', '', data1)
data1 = data1.replace('\n', '')

data2 = data2.lower()
data2 = re.sub('^Section [1-9].', '', data2)
data2 = re.sub(r'==.*?==+', '', data2)
data2 = re.sub(r'CHAPTER \d+', '', data2)
data2 = re.sub('[\(\[].*?[\)\]]', '', data2)
data2 = re.sub(r'[^a-zA-Z0-9\s]', '', data2)
data2 = data2.replace('\n', '')

# tokenizing
token1 = nltk.word_tokenize(data1)
token2 = nltk.word_tokenize(data2)

# frequency Distribution
fdist1 = FreqDist(token1)
print(fdist1.most_common(20))
fdist2 = FreqDist(token2)
print(fdist2.most_common(20))

fig = plt.figure(figsize=(40, 30))
# plotting freq. Dist. of 20 most common words
fdist1.plot(20)
fdist2.plot(20)

# Plotting the relationship between
# word length and word frequency
# before removing stop words
plotRelationShip(FreqDist(token1))
plotRelationShip(FreqDist(token2))

# removing stopwords
# Finding Stopwords
stop_words = set(stopwords.words('english'))

# stopwords in data 1
sp1 = []
rem1 = []
for w in token1:
    if w in stop_words:
        sp1.append(w)
    else:
	rem1.append(w)
			 	

# stopwords in data 2
sp2 = []
rem2 = []
for w in token2:
    if w in stop_words:
        sp2.append(w)
    else:
	rem1.append(w)

# Plotting the relationship between
# word length and word frequency
# after removing stop words

plotRelationShip(FreqDist(rem1))
plotRelationShip(FreqDist(rem2))
	
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
tags1 = [ t for (w,t) in tagged1]
freq_dist(tags1)
tagged2 = nltk.pos_tag(word_list2)
tags2 = [ t for (w,t) in tagged2]
freq_dist(tags2)
