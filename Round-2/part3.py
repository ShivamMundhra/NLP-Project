import nltk
import urllib.request
import urllib.parse
import urllib.error
import ssl
import json
import re

# import nltk
# nltk.download('maxent_ne_chunker')
# nltk.download('words')


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

# tokenizing
print('Tokenizing the texts of the books')
token1 = nltk.word_tokenize(data1)
token2 = nltk.word_tokenize(data2)
print('Tokenizing done')
print('--------------------------------------------------------------------------------------------------')

# pos tagging
print('POS tagging')
tagged1 = nltk.pos_tag(token1)
tagged2 = nltk.pos_tag(token2)
print('POS tagging done')
print('--------------------------------------------------------------------------------------------------')

# NER
print('Named entity recognition')
namedEnt1 = nltk.ne_chunk(tagged1)
namedEnt2 = nltk.ne_chunk(tagged2)
print('Named entity recognition done')
print('--------------------------------------------------------------------------------------------------')

# extracting different kinds of relations
print('Extracting different kinds of relations')
SON = re.compile(r'.*\bson\b')
DAUGHTER = re.compile(r'.*\bdaughter\b')
BROTHER = re.compile(r'.*\bbrother\b')
WIFE = re.compile(r'.*\bswife\b')

print('--------------------------------------SON------------------------------------------------------------')
for rel in nltk.sem.extract_rels('PER', 'PER', namedEnt1, corpus='ace', pattern=SON):
    print(nltk.sem.rtuple(rel))

print('------------------------------------DAUGHTER--------------------------------------------------------------')
for rel in nltk.sem.extract_rels('PER', 'PER', namedEnt1, corpus='ace', pattern=DAUGHTER):
    print(nltk.sem.rtuple(rel))

print('------------------------------------BROTHER--------------------------------------------------------------')
for rel in nltk.sem.extract_rels('PER', 'PER', namedEnt1, corpus='ace', pattern=BROTHER):
    print(nltk.sem.rtuple(rel))

print('--------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------')

print('--------------------------------------SON------------------------------------------------------------')
for rel in nltk.sem.extract_rels('PER', 'PER', namedEnt2, corpus='ace', pattern=SON):
    print(nltk.sem.rtuple(rel))

print('------------------------------------DAUGHTER--------------------------------------------------------------')
for rel in nltk.sem.extract_rels('PER', 'PER', namedEnt2, corpus='ace', pattern=DAUGHTER):
    print(nltk.sem.rtuple(rel))

print('------------------------------------BROTHER--------------------------------------------------------------')
for rel in nltk.sem.extract_rels('PER', 'PER', namedEnt2, corpus='ace', pattern=BROTHER):
    print(nltk.sem.rtuple(rel))

print('--------------------------------------------------------------------------------------------------')
