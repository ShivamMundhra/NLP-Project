from nltk.corpus import wordnet
import nltk
import urllib.request
import urllib.parse
import urllib.error
import ssl
import json
import nltk
import re
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

# nltk.download('wordnet')
lemmatizer = nltk.WordNetLemmatizer()


def get_word_postag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('N'):
        return wordnet.NOUN
    else:
        return wordnet.NOUN


def normalise(word, tag):
    # Normalises words to lowercase and stems and lemmatizes it
    word = word.lower()
    postag = get_word_postag(tag)
    word = lemmatizer.lemmatize(word, postag)
    return word


def removePrefix(text, prefix):
    return text[len(prefix):]


def plot(X, Y, xlabel, ylabel, title):
    plt.bar(X, Y, tick_label=X, width=0.8, color=['red', 'green'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()


def plotRelationShip(fd, prefix):

    data = {}

    for i in fd.keys():
        ii = removePrefix(i, prefix)
        if ii not in data.keys():
            data[ii] = fd[i]
        else:
            data[ii] += fd[i]

    X = []
    Y = []

    for i in data.keys():
        X.append(i)

    for i in X:
        Y.append(data[i])

    # Plotting a bar graph for recorded data
    xlabel = 'Category/Supersense'
    ylabel = 'Frequency'
    title = 'Relationship between categories and frequency'
    plot(X, Y, xlabel, ylabel, title)


print('--------------------------------------------------------------------------------------------------')
print('Downloading books')
site1 = 'http://www.gutenberg.org//cache/epub/7864/pg7864.txt'
print('Extracting ', site1)
uh1 = urllib.request.urlopen(site1)
data1 = uh1.read().decode('utf8')

# Extracting second text
site2 = 'https://www.gutenberg.org//cache/epub/22381/pg22381.txt'
print('Extracting ', site2)
uh2 = urllib.request.urlopen(site2)
data2 = uh2.read().decode('utf8')
print('Downloading of books complete')
print('--------------------------------------------------------------------------------------------------')

# pre-processing the text of Book-1
print('Preprocessing text of both books')
data1 = data1.lower()
data1 = re.sub('^Section [1-9].', '', data1)
data1 = re.sub(r'==.*?==+', '', data1)
data1 = re.sub(r'CHAPTER \d+', '', data1)
data1 = re.sub('[\(\[].*?[\)\]]', '', data1)
data1 = re.sub(r'[^a-zA-Z0-9\s]', '', data1)
data1 = data1.replace('\n', '')

# pre-processing the text of Book-2
data2 = data2.lower()
data2 = re.sub('^Section [1-9].', '', data2)
data2 = re.sub(r'==.*?==+', '', data2)
data2 = re.sub(r'CHAPTER \d+', '', data2)
data2 = re.sub('[\(\[].*?[\)\]]', '', data2)
data2 = re.sub(r'[^a-zA-Z0-9\s]', '', data2)
data2 = data2.replace('\n', '')
print('Preprocessing done')
print('--------------------------------------------------------------------------------------------------')

# tokenizing
print('Tokenizing the texts of the books')
token1 = nltk.word_tokenize(data1)
token2 = nltk.word_tokenize(data2)
print('Tokenizing done')
print('--------------------------------------------------------------------------------------------------')

# Finding and removing stopwords
print('Removing stopwords')
stop_words = set(stopwords.words('english'))

# stopwords in data 1
word_list1 = []
for w in token1:
    if not w in stop_words:
        word_list1.append(w)

word_list2 = []
for w in token2:
    if not w in stop_words:
        word_list2.append(w)

print('Stopwords removed')
print('--------------------------------------------------------------------------------------------------')

# extracting nouns and verbs using pos tagging
print('Extracting nouns and verbs using POS tagging')
tagged1 = nltk.pos_tag(word_list1)
n1 = set()
v1 = set()
for word, tag in tagged1:
    word = normalise(word, tag)
    if tag.startswith('N'):
        n1.add(word)
    if tag.startswith('V'):
        v1.add(word)

tagged2 = nltk.pos_tag(word_list2)
n2 = set()
v2 = set()
for word, tag in tagged2:
    word = normalise(word, tag)
    if tag.startswith('N'):
        n2.add(word)
    if tag.startswith('V'):
        v2.add(word)

nouns1 = list(n1)
verbs1 = list(v1)
nouns2 = list(n2)
verbs2 = list(v2)
print('All nouns and verbs extracted')
print('--------------------------------------------------------------------------------------------------')

# finding the category of each noun and verb and plotting frequency distribution
print('Finding the category of each noun and verb using wordnet and plotting frequency distribution')
print('Plot for nouns of book-1:')
lst = []
for word in nouns1:
    syn = wordnet.synsets(word, pos=wordnet.NOUN)
    if len(syn) > 0:
        lst.append(syn[0].lexname())

plotRelationShip(FreqDist(lst), 'noun.')

print('Plot for verbs of book-1:')
lst = []
for word in verbs1:
    syn = wordnet.synsets(word, pos=wordnet.VERB)
    if len(syn) > 0:
        lst.append(syn[0].lexname())

plotRelationShip(FreqDist(lst), 'verb.')

print('Plot for nouns of book-2:')
lst = []
for word in nouns2:
    syn = wordnet.synsets(word, pos=wordnet.NOUN)
    if len(syn) > 0:
        lst.append(syn[0].lexname())

plotRelationShip(FreqDist(lst), 'noun.')

print('Plot for verbs of book-2:')
lst = []
for word in verbs2:
    syn = wordnet.synsets(word, pos=wordnet.VERB)
    if len(syn) > 0:
        lst.append(syn[0].lexname())

plotRelationShip(FreqDist(lst), 'verb.')
print('--------------------------------------------------------------------------------------------------')
