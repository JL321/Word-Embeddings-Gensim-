from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from random import shuffle
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import random
from random import shuffle

dataSet = pd.read_csv("imdb_tr.csv", header = None, encoding = "ISO-8859-1")
dataSet = dataSet.iloc[:, [1,2]]
dataSetA = np.array(dataSet).tolist()
dataSetA = dataSetA[1:5001]

shuffle(dataSetA)

featureSet = []
labelSet = []

listFeature = []

for i in range(5000):
    featureSet.append(dataSetA[i][0])
    labelSet.append(dataSetA[i][1])

translate = str.maketrans('','',string.punctuation)

flag = True

counter = 0
dict_count = 0
dict_conv = {}
max_sentence = 0

for sentCount,sentence_s in enumerate(featureSet):
    new_sentence = []
    sentence_s = sentence_s.lower()
    
    while flag:
        if sentence_s[counter] == '<':
            if sentence_s[counter:counter+6] == '<br />':
                sentence_s = sentence_s[:counter]+sentence_s[counter+6:]
                counter -= 1
        counter += 1
        if counter >= len(sentence_s)-6:
            flag = False
    #print(sentence_s)
    sentence = sentence_s.translate(translate)
    
    counter = 0
    
    flag = True
    
    sentence = sentence.split()
    
    word_c = 0
    
    for i in range(len(sentence)):
        v = i - word_c
        if sentence[v] in stopwords.words('english'):
            del sentence[v]
            word_c += 1
        v = i - word_c
        if sentence[v] not in dict_conv:
            dict_conv[sentence[v]] = dict_count
            dict_count += 1
    
    if len(sentence) > max_sentence:
        max_sentence = len(sentence)
    
    #new_sentence = ' '.join(sentence)
    
    featureSet[sentCount] = sentence        

minV = 10

for i in featureSet:
    if len(i) < minV:
        minV = len(i)

print('Testing')
        
print(minV)

print(featureSet)

print('Cycle 1 complete')

model = Word2Vec(featureSet, size = 50, window = 4, sg = 1, min_count = 10)

#model.train(featureSet, total_examples = len(featureSet), epochs = 5)

model.save('word2vec.model')

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

wvm = Word2Vec.load('word2vec.model')

vocab = list(wvm.wv.vocab)

X = wvm[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

df  = pd.DataFrame(X_tsne, index = vocab, columns = ['x','y'])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.scatter(df['x'],df['y'])

ax.set_xlim(-10,10)
ax.set_ylim(-10,10)

for word,pos in df.iterrows():
    ax.annotate(word, pos)
    
plt.show()

