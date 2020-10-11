import urllib.request, urllib.parse, urllib.error
import ssl
import json
import nltk
import re
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

CONTRACTION_MAP = {
	"ain't": "is not",
	"aren't": "are not",
	"can't": "cannot",
	"can't've": "cannot have",
	"'cause": "because",
	"could've": "could have",
	"couldn't": "could not",
	"couldn't've": "could not have",
	"didn't": "did not",
	"doesn't": "does not",
	"don't": "do not",
	"hadn't": "had not",
	"hadn't've": "had not have",
	"hasn't": "has not",
	"haven't": "have not",
	"he'd": "he would",
	"he'd've": "he would have",
	"he'll": "he will",
	"he'll've": "he he will have",
	"he's": "he is",
	"how'd": "how did",
	"how'd'y": "how do you",
	"how'll": "how will",
	"how's": "how is",
	"I'd": "I would",
	"I'd've": "I would have",
	"I'll": "I will",
	"I'll've": "I will have",
	"I'm": "I am",
	"I've": "I have",
	"i'd": "i would",
	"i'd've": "i would have",
	"i'll": "i will",
	"i'll've": "i will have",
	"i'm": "i am",
	"i've": "i have",
	"isn't": "is not",
	"it'd": "it would",
	"it'd've": "it would have",
	"it'll": "it will",
	"it'll've": "it will have",
	"it's": "it is",
	"let's": "let us",
	"ma'am": "madam",
	"mayn't": "may not",
	"might've": "might have",
	"mightn't": "might not",
	"mightn't've": "might not have",
	"must've": "must have",
	"mustn't": "must not",
	"mustn't've": "must not have",
	"needn't": "need not",
	"needn't've": "need not have",
	"o'clock": "of the clock",
	"oughtn't": "ought not",
	"oughtn't've": "ought not have",
	"shan't": "shall not",
	"sha'n't": "shall not",
	"shan't've": "shall not have",
	"she'd": "she would",
	"she'd've": "she would have",
	"she'll": "she will",
	"she'll've": "she will have",
	"she's": "she is",
	"should've": "should have",
	"shouldn't": "should not",
	"shouldn't've": "should not have",
	"so've": "so have",
	"so's": "so as",
	"that'd": "that would",
	"that'd've": "that would have",
	"that's": "that is",
	"there'd": "there would",
	"there'd've": "there would have",
	"there's": "there is",
	"they'd": "they would",
	"they'd've": "they would have",
	"they'll": "they will",
	"they'll've": "they will have",
	"they're": "they are",
	"they've": "they have",
	"to've": "to have",
	"wasn't": "was not",
	"we'd": "we would",
	"we'd've": "we would have",
	"we'll": "we will",
	"we'll've": "we will have",
	"we're": "we are",
	"we've": "we have",
	"weren't": "were not",
	"what'll": "what will",
	"what'll've": "what will have",
	"what're": "what are",
	"what's": "what is",
	"what've": "what have",
	"when's": "when is",
	"when've": "when have",
	"where'd": "where did",
	"where's": "where is",
	"where've": "where have",
	"who'll": "who will",
	"who'll've": "who will have",
	"who's": "who is",
	"who've": "who have",
	"why's": "why is",
	"why've": "why have",
	"will've": "will have",
	"won't": "will not",
	"won't've": "will not have",
	"would've": "would have",
	"wouldn't": "would not",
	"wouldn't've": "would not have",
	"y'all": "you all",
	"y'all'd": "you all would",
	"y'all'd've": "you all would have",
	"y'all're": "you all are",
	"y'all've": "you all have",
	"you'd": "you would",
	"you'd've": "you would have",
	"you'll": "you will",
	"you'll've": "you will have",
	"you're": "you are",
	"you've": "you have"
}

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

# pre-processing the text of Book-1
data1 = data1.lower()
data1 = re.sub('^Section [1-9].', '', data1)
contractions_re = re.compile('(%s)' % '|'.join(CONTRACTION_MAP.keys()))
def replace(match):
    return CONTRACTION_MAP[match.group(0)]
data1 = contractions_re.sub(replace, data1)
data1 = re.sub(r'==.*?==+', '', data1)
data1 = re.sub(r'CHAPTER \d+', '', data1)
data1 = re.sub('[\(\[].*?[\)\]]', '', data1)
data1 = re.sub(r'[^a-zA-Z0-9\s]', '', data1)
data1 = data1.replace('\n', '')

# pre-processing the text of Book-2
data2 = data2.lower()
data2 = re.sub('^Section [1-9].', '', data2)
contractions_re = re.compile('(%s)' % '|'.join(CONTRACTION_MAP.keys()))
def replace(match):
    return CONTRACTION_MAP[match.group(0)]
data2 = contractions_re.sub(replace, data2)
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

# plotting freq. Dist. of 20 most common words
fig = plt.figure(figsize=(40, 30))
fdist1.plot(20)
fdist2.plot(20)

# Plotting the relationship between word length and word frequency before removing stop words
plotRelationShip(FreqDist(token1))
plotRelationShip(FreqDist(token2))

# Finding and removing stopwords
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

# Plotting the relationship between word length and word frequency after removing stop words
plotRelationShip(FreqDist(rem1))
plotRelationShip(FreqDist(rem2))
	
# Creating Word Cloud before removing the stopwords

# for data 1
wordcloud1 = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', stopwords = [], collocations=False).generate(data1)
plt.figure(figsize=(40, 30))
plt.imshow(wordcloud1, interpolation = 'bilinear')
plt.axis("off");

# for data 2
wordcloud2 = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', stopwords = [], collocations=False).generate(data2)
plt.figure(figsize=(40, 30))
plt.imshow(wordcloud2)
plt.axis("off");

# Creating Word Cloud after removing stopwords

# for data 1
modified_wordcloud1 = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', stopwords = sp1, collocations = False).generate(data1)
plt.figure(figsize=(40, 30))
plt.imshow(modified_wordcloud1)
plt.axis("off");

# for data 2
modified_wordcloud2 = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', stopwords = sp2, collocations = False).generate(data2)
plt.figure(figsize=(40, 30))
plt.imshow(modified_wordcloud2)
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
