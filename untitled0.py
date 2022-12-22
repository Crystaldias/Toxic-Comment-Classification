# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:51:45 2018

@author: Granger
"""
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import time 
#from gensim.models import Word2Vec 
import multiprocessing # for threading of word2vec model process
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from keras.layers import LSTM,Activation, Dense, Dropout, Input, Embedding, Flatten 
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
'''
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
'''
#train = pd.read_csv(r'F:\6th sem\Minor project\train.csv')
#test = pd.read_csv(r'F:\6th sem\Minor project\test.csv')

total= pd.read_csv(r'D:\Minor project\train.csv')
X=total.comment_text
y=total.score
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(X_train.head(5),y_train.head(5))
print(X_test.head(5),y_test.head(5))

"""
#Plotting the data
x=total.iloc[:,2:].sum()
ax=sns.barplot(x.index, x.values,alpha=1.0)
print(ax)
plt.title("# per class")
plt.ylabel("# of occurrences",fontsize=12)
plt.xlabel("Type",fontsize=12)

#adding the text labels;
rects=ax.patches
labels=x.values
for rect, label in zip(rects,labels):
    height=rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2,height+5,label, ha='center', va='bottom')
"""
#Storing the target in a variable
target_labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# Contraction replacement patterns
def clean_text(text):
    text = re.sub(r"what's","what is ",text)
    text = re.sub(r"\'s"," ",text)
    text = re.sub(r"\'ve"," have ",text)
    text = re.sub(r"can't","cannot ",text)
    text = re.sub(r"n't"," not ",text)
    text = re.sub(r"i'm","i am ",text)
    text = re.sub(r"\'re"," are ",text)
    text = re.sub(r"\'d"," would ", text)
    text = re.sub(r"\'ll", " will ",text)
    text = re.sub(r"\ 'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

#Apply the above function to both train and  test dataset 
cleaned_train_comment = []
for i in range(0,len(X_train)):
    cleaned_comment = clean_text(total['comment_text'][i])
    cleaned_train_comment.append(cleaned_comment)
total['comment_text'] = pd.Series(cleaned_train_comment).astype(str)


"""
cleaned_test_comment = []
for i in range(0,len(X_test)):
    cleaned_comment = clean_text(X_test['comment_text'][i])
    cleaned_test_comment.append(cleaned_comment)
X_test['comment_text'] = pd.Series(cleaned_test_comment).astype(str)
"""

#Split the train and test dataset
#X=train.comment_text
#test_X=test.comment_text

"""
#word2vec

# hyper parameters of the word2vec model
num_features = 300 # dimensions of each word embedding
min_word_count = 1 # this is not advisable but since we need to extract
# feature vector for each word we need to do this
#num_workers = multiprocessing.cpu_count() # number of threads running in parallel
context_size = 7 # context window length
downsampling = 1e-3 # downsampling for very frequent words
seed = 1 # seed for random number generator to make results reproducible


word2vec = Word2Vec(
    sg = 1, seed = seed,
    size = num_features,
    min_count = min_word_count,
    window = context_size,
    sample = downsampling
)


#word2vec = Word2Vec()
#buildvolabulary
word=word2vec.build_vocab(X)
#print(vect[''])
#save model
#word2vec.save('model.bin')
#load model
#new_model = word2vec.load('model.bin')
X_dtm = word2vec.train(word, total_examples = word2vec.corpus_count, epochs = 100)

#print(X_dtm)



#Transform the test data into a document-term matrix
test_X_dtm = word2vec.train(test_X)
"""

"""
#tf-idf
#Transform text to feature vectors
vect=TfidfVectorizer(stop_words='english',ngram_range=(1,2), max_df=0.5, min_df=2)

#Learn the vocabulary from the feature vectors , then use it to create a document-term matrix
X_dtm= vect.fit_transform(X_train)
#print(X_dtm)
#vect.get_feature_names()
#Transform the test data into a document-term matrix
test_X_dtm = vect.transform(X_test)
"""

"""
#count-vectorizer
#Transform text to feature vectors  stop_words='english',ngram_range=(1,2), max_df=0.5, min_df=2
vect=CountVectorizer(stop_words='english',ngram_range=(1,2), max_df=0.5, min_df=2)

#vect.get_feature_names()
#Learn the vocabulary from the feature vectors , then use it to create a document-term matrix
X_dtm= vect.fit_transform(X_train)
print(X_dtm)
#vect.get_feature_names()

#Transform the test data into a document-term matrix
test_X_dtm = vect.transform(X_test)
"""





#svm  accuracy ~ .95
#model = svm.LinearSVC()



"""
#KNN memory error
model = KNeighborsClassifier(n_neighbors=5) # default value for n_neighbors is 5
"""


"""
#decission tree error: can't handle mix of binary and continuous
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini 
model = tree.DecisionTreeRegressor() #for regression
"""


"""
#random forest 
model= RandomForestClassifier()
"""


#logistic regression
#model = LogisticRegression(C=6.0)


#LSTM

max_words = 20000
max_len = 1354
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=None)
    

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,64,input_length=max_len)(inputs)
    layer = LSTM(50)(layer)
    layer = Dense(50,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
batch_size = 32
epochs = 1
model.fit(sequences_matrix,y_train,batch_size=batch_size,epochs=epochs, validation_split=0.1)
prediction_train= model.predict(sequences_matrix)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen = max_len)

accr = model.evaluate(test_sequences_matrix, y_test)
print('Test set\n Loss: {:0.3f}\n Accuracy: {:0.3f}'.format(accr[0],accr[1]))

#print(prediction_train)
#print('Training accuracy is {}'.format(accuracy_score(y_train,prediction_train)))
#print( "test accuracy is {}".format(model.score(test_X_dtm,y_test)))

#a=f1_score(y_train,prediction_train, average = 'macro', labels=np.unique(prediction_train))
#print(a)

"""
max_prob = 0
#test_comment=vect.transform([input("Enter the comment : ")])

for label in target_labels:
    print('...Processing{} '.format(label))
    #with timer(b):
    y = total[label]
    print(y)
    #train the model using X_dtm & y
    model.fit(X_dtm,y)

    #compute the training accuracy
    y_pred_X = model.predict(X_dtm)
    print('Training accuracy is {}'.format(accuracy_score(y,y_pred_X)))
    #compute the predict probabilities for X_test_dtm
    test_y_prob = model.predict_proba(test_X_dtm)[:,1]
    print(test_y_prob)
    if(max_prob<test_y_prob[0][1]):
        max_prob = test_y_prob[0][1]
    if (max_prob == test_y_prob[0][1]):
        a = label
        
    print(test_y_prob)


print(max_prob)
print("Result : " + a)


test_comment=vect.transform([input("Enter the comment")])
abc = []
maxval = 0
for i in range(6):
    abc.append(model.predict(test_comment))
print(abc)
"""