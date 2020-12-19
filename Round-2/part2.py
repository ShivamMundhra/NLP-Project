import nltk
import urllib.request
import urllib.parse
import urllib.error
import ssl
import json
import pandas as pd
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import numpy as np

# import nltk
# nltk.download('maxent_ne_chunker')
# nltk.download('words')


def plot(X, Y, xlabel, ylabel, title):
    plt.bar(X, Y, tick_label=X, width=0.8, color=['red', 'blue'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()
    plt.subplots_adjust(bottom=0.18)


def plotRelationShip(fd):

    data = {}

    for i in fd.keys():
        if i not in data.keys():
            data[i] = fd[i]
        else:
            data[i] += fd[i]

    X = []
    Y = []

    for i in data.keys():
        X.append(i)

    for i in X:
        Y.append(data[i])

    # Plotting a bar graph for recorded data
    xlabel = 'Entities'
    ylabel = 'Frequency'
    title = 'Relationship between entities and frequency'
    plot(X, Y, xlabel, ylabel, title)


def metrics(truth, run):
    t = set(truth)
    r = set(run)
    intersection = r & t
    True_positive = float(len(intersection))
    if float(len(run)) >= float(True_positive):
        False_positive = len(run) - True_positive
    else:
        False_positive = True_positive - len(run)
    True_negative = 0
    if len(truth) >= len(run):
        False_negative = len(truth) - len(run)
    else:
        False_negative = 0
    accuracy = (float(True_positive) + float(True_negative))/(float(True_positive) +
                                                              float(True_negative) + float(False_positive) + float(False_negative))
    precision = float(True_positive) / \
        (float(True_positive) + float(False_positive))
    recall = float(True_positive) / \
        (float(True_positive) + float(False_negative))
    F_measure = (2 * recall * precision) / (recall + precision)
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F-measure: ", F_measure)
    d = {'Predicted Negative': [True_negative, False_negative],
         'Predicted Positive': [False_positive, True_positive]}
    metricsdf = pd.DataFrame(d, index=['Negative Cases', 'Positive Cases'])
    return metricsdf

# def metrics(truth, run):
#     truth = truth
#     run = run
#     TP = float(len(set(run) & set(truth)))
#     if float(len(run)) >= float(TP):
#         FP = len(run) - TP
#     else:
#         FP = TP - len(run)
#     TN = 0
#     if len(truth) >= len(run):
#         FN = len(truth) - len(run)
#     else:
#         FN = 0
#     accuracy = (float(TP)+float(TN))/float(len(truth))
#     recall = (float(TP))/float(len(truth))
#     precision = float(TP)/(float(FP)+float(TP))
#     print("The accuracy is %r" % accuracy)
#     print("The recall is %r" % recall)
#     print("The precision is %r" % precision)
#     d = {'Predicted Negative': [TN, FN], 'Predicted Positive': [FP, TP]}
#     metricsdf = pd.DataFrame(d, index=['Negative Cases', 'Positive Cases'])
#     return metricsdf


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

# plotting entities vs. frequency charts
print('Plotting entities vs. frequency chart for book-1')
lst = []
seen = set()
for chunk in namedEnt1:
    if hasattr(chunk, 'label'):
        # print(chunk.label(), ' '.join(c[0] for c in chunk))
        tmp = ' '.join(c[0] for c in chunk)
        if tmp not in seen:
            seen.add(tmp)
            lst.append(chunk.label())

plotRelationShip(FreqDist(lst))

lst = []
seen = set()
print('Plotting entities vs. frequency chart for book-2')
for chunk in namedEnt2:
    if hasattr(chunk, 'label'):
        # print(chunk.label(), ' '.join(c[0] for c in chunk))
        tmp = ' '.join(c[0] for c in chunk)
        if tmp not in seen:
            seen.add(tmp)
            lst.append(chunk.label())

plotRelationShip(FreqDist(lst))
print('--------------------------------------------------------------------------------------------------')


print('Running NER on random paragraphs from books to evaluate the algorithm')
print('')
print('Paragraph-1 :')
data1 = "Samvarana begat upon his wife, Tapati, the daughter of Surya, a son named Kuru. This Kuru was exceedingly virtuous, and therefore, he was installed on the throne by his people. It is after his name that the field called Kuru-jangala has become so famous in the world. Devoted to asceticism, he made that field (Kurukshetra) sacred by practising asceticism there. And it has been heard by us that Kuru's highly intelligent wife, Vahini, brought forth five sons, viz., Avikshit, Bhavishyanta, Chaitraratha, Muni and the celebrated Janamejaya. And Avikshit begat Parikshit the powerful, Savalaswa, Adhiraja, Viraja, Salmali of great physical strength, Uchaihsravas, Bhangakara and Jitari the eighth. In the race of these were born, as the fruit of their pious acts seven mighty car-warriors with Janamejaya at their head. And unto Parikshit were born sons who were all acquainted with (the secrets of) religion and profit. And they were named Kakshasena and Ugrasena, and Chitrasena endued with great energy, and Indrasena and Sushena and Bhimasena. And the sons of Janamejaya were all endued with great strength and became celebrated all over the world. And they were Dhritarashtra who was the eldest, and Pandu and Valhika, and Nishadha endued with great energy, and then the mighty Jamvunada, and then Kundodara and Padati and then Vasati the eighth. And they were all proficient in morality and profit and were kind to all creatures. Among them Dhritarashtra became king. And Dhritarashtra had eight sons, viz., Kundika, Hasti, Vitarka, Kratha the fifth, Havihsravas, Indrabha, and Bhumanyu the invincible, and Dhritarashtra had many grandsons, of whom three only were famous. They were, O king, Pratipa, Dharmanetra, Sunetra. Among these three, Pratipa became unrivalled on earth. And, O bull in Bharata's race, Pratipa begat three sons, viz., Devapi, Santanu, and the mighty car-warrior Valhika. The eldest Devapi adopted the ascetic course of life, impelled thereto by the desire of benefiting his brothers. And the kingdom was obtained by Santanu and the mighty car-warrior Valhika."
token = nltk.word_tokenize(data1)
tagged = nltk.pos_tag(token)
chunked = nltk.ne_chunk(tagged)

run = []
truth = ['Samvarana', 'Tapati', 'Surya', 'Kuru', 'Vahini', 'Avikshit', 'Bhavishyanta', 'Chaitraratha', 'Muni', 'Janamejaya', 'Parikshit', 'Savalaswa', 'Adhiraja', 'Viraja', 'Salmali', 'Uchaihsravas', 'Bhangakara', 'Jitari', 'Kakshasena', 'Ugrasena', 'Chitrasena', 'Indrasena',
         'Sushena', 'Bhimasena', 'Dhritarashtra', 'Pandu', 'Valhika', 'Nishadha', 'Jamvunada', 'Kundodara', 'Padati', 'Vasati', 'Kundika', 'Hasti', 'Vitarka', 'Kratha', 'Havihsravas', 'Indrabha', 'Bhumanyu', 'Pratipa', 'Dharmanetra', 'Sunetra', 'Bharata', 'Devapi', 'Santanu']
for chunk in chunked:
    if hasattr(chunk, 'label'):
        if chunk.label() == 'PERSON':
            # print(chunk.label(), ' '.join(c[0] for c in chunk))
            ne = ' '.join(c[0] for c in chunk)
            run.append(ne)

print('Maunally labelled PERSON entities : ')
print(set(truth))
print('')
print('Algorithm labelled PERSON entities : ')
print(set(run))
print('')
print('Evaluation :')
print(metrics(truth, run))

print('\n')

print('Paragraph-2 :')
data2 = "Equipped with the magic helmet and wallet, and armed with a sickle, the gift of Hermes, he attached to his feet the winged sandals, and flew to the abode of the Gorgons, whom he found fast asleep. Now as Perseus had been warned by his celestial guides that whoever looked upon these weird sisters would be transformed into stone, he stood with averted face before the sleepers, and caught on his bright metal shield their triple image. Then, guided by Pallas-Athene, he cut off the head of the Medusa, which he placed in his wallet. No sooner had he done so than from the headless trunk there sprang forth the winged steed Pegasus, and Chrysaor, the father of the winged giant Geryon. He now hastened to elude the pursuit of the two surviving sisters, who, aroused from their slumbers, eagerly rushed to avenge the death of their sister."
token = nltk.word_tokenize(data2)
tagged = nltk.pos_tag(token)
chunked = nltk.ne_chunk(tagged)

run = []
truth = ['Hermes', 'Perseus', 'Medusa', 'Chrysaor', 'Geryon']
for chunk in chunked:
    if hasattr(chunk, 'label'):
        if chunk.label() == 'PERSON':
            # print(chunk.label(), ' '.join(c[0] for c in chunk))
            ne = ' '.join(c[0] for c in chunk)
            run.append(ne)

print('Maunally labelled PERSON entities : ')
print(set(truth))
print('')
print('Alogorithm labelled PERSON entities : ')
print(set(run))
print('')
print('Evaluation :')
print(metrics(truth, run))
print('--------------------------------------------------------------------------------------------------')

# print([" ".join(w+" "+t for w, t in elt)
#        for elt in chunked if isinstance(elt, nltk.Tree)])
