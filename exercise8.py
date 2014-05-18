from __future__ import print_function

import nltk
import string
import os
import pdb
import logging
import numpy as np
import sys
import pylab as pl

from time import time
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans, Ward, SpectralClustering
from bs4 import BeautifulSoup


print('================================ STARTING ================================')

path = os.path.join(sys.path[0], "reuters21578-xml/")
token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in ['reuters', 'reuter', 'blah']]
    stems = stem_tokens(tokens, stemmer)
    return stems

soups = []

for i in xrange(0,10):
    print("Parsing reut2-00%d" % i)
    soups.append(BeautifulSoup(open(path + 'reut2-00' + str(i) + '.xml'),"xml"))

for i in xrange(10,22):
    print("Parsing reut2-0%d" % i)
    soups.append(BeautifulSoup(open(path + 'reut2-0' + str(i) + '.xml'),"xml"))    

any_in = lambda a, b: any(t in b for t in a)
toptopics = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'corn', 'wheat']
categories = toptopics

topics_dict = {}
orig_topics_dict = {}

train_bodies = []
test_bodies = []
all_bodies = []
y_train = []
y_test = []
labels = []

bigtopicslist = []
alltopics = []

for i in xrange(len(soups)):
    for topics in soups[i].find_all('TOPICS'):
        bigtopicslist.append(topics.find_all('D'))

for i in xrange(len(bigtopicslist)):
    for j in xrange(len(bigtopicslist[i])):
        bigtopicslist[i][j] = str(bigtopicslist[i][j].text)
        if(bigtopicslist[i][j] not in alltopics):
            alltopics.append(bigtopicslist[i][j])
    orig_topics_dict.update({i+1:bigtopicslist[i]})
    bigtopicslist[i] = [t for t in bigtopicslist[i] if t in toptopics]
    topics_dict.update({i+1:bigtopicslist[i]})


for i in xrange(len(soups)):
    topicslist = []
    for header in soups[i].find_all('REUTERS'):
        newid = int(header.get('NEWID'))
        topicslist = orig_topics_dict[newid]
        split = str(header.get('LEWISSPLIT')) 
        if (header.has_attr('BODY')):
            body = header.BODY
        else:
            body = header.TEXT
        all_bodies.append(body)
        labels.append(topicslist)
        if (len(topicslist)==1 and any_in(topicslist, toptopics)):
            if (split == 'TRAIN'):
                train_bodies.append(body)
                y_train.append(topicslist[0])
            else: # split == 'TEST'
                test_bodies.append(body)
                y_test.append(topicslist[0])

data_train = []
data_test = []
data_all = []

for i in xrange(len(train_bodies)):
    try:
        temp = train_bodies[i]
        temp = str(temp.text)
        temp = temp.lower()
        temp = temp.translate(None, string.punctuation)
        data_train.append(temp)
    except:
        print(i)
        print('TRAIN ERROR')
        continue

for i in xrange(len(test_bodies)):
    try:
        temp = test_bodies[i]
        temp = str(temp.text)
        temp = temp.lower()
        temp = temp.translate(None, string.punctuation)
        data_test.append(temp)
    except:
        print(i)
        print('TEST ERROR')
        continue

for i in xrange(len(all_bodies)):
    try:
        temp = all_bodies[i]
        temp = str(temp.text)
        temp = temp.lower()
        temp = temp.translate(None, string.punctuation)
        data_all.append(temp)
    except:
        print(i)
        print('ALL ERROR')
        continue

###########################################
# TOPICS count
#
# vectorizer = CountVectorizer(min_df=1, vocabulary=toptopics)
# X_train = vectorizer.fit_transform(data_train)    
#
#
###########################################


print("About to TFIDF.")

vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
X_train = vectorizer.fit_transform(data_train)
X = X_train
X_all = vectorizer.transform(data_all)

    

print("Extracting features from the test dataset using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test)
duration = time() - t0
data_test_size_mb = sys.getsizeof(data_test) / 1024
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# mapping from integer feature name to original token string
feature_names = np.asarray(vectorizer.get_feature_names())



###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                try:
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
                except:
                    continue                        
        print()

        print("classification report:")
        print("accuracy:    %0.5f" % metrics.accuracy_score(y_test, pred))
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC()
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC()))


# make some plots
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

pl.figure(figsize=(12,8))
pl.title("Score")
pl.barh(indices, score, .2, label="score", color='r')
pl.barh(indices + .3, training_time, .2, label="training time", color='g')
pl.barh(indices + .6, test_time, .2, label="test time", color='b')
pl.yticks(())
pl.subplots_adjust(left=.25)
pl.subplots_adjust(top=.95)
pl.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    pl.text(-.3, i, c)

"""
if (1):
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    lsa = TruncatedSVD(1000)
    X = lsa.fit_transform(X_all)
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    X_all = Normalizer(copy=False).fit_transform(X)
    print("done in %fs" % (time() - t0))
    print()
"""

print("Extracting 1000 best features by a chi-squared test")
t0 = time()
ch2 = SelectKBest(chi2, k=1000)
X_all = ch2.fit_transform(X_all, labels)
print("done in %fs" % (time() - t0))
print()


###############################################################################
# Do the actual clustering

true_k = len(alltopics)

for i in xrange(3):
    print('=' * 80)
    if(i == 0):
        print("MiniBatchKMeans")
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=True)
        t0 = time()
        km.fit(X_all)
        print("Clustering sparse data with %s" % km)
        print("done in %0.3fs" % (time() - t0))
        print()
        
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
        print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
        
    elif(i == 1):
        print("KMeans")
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=True)
        t0 = time()
        km.fit(X_all)
        print("Clustering sparse data with %s" % km)
        print("done in %0.3fs" % (time() - t0))
        print()

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
        print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))

    else:
        print("Ward")
        km = Ward()
        print("Clustering sparse data with %s" % km)
        t0 = time()
        km.fit(X_all.toarray())

        print("done in %0.3fs" % (time() - t0))
        print()

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
        print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
       
print()

print('================================ FINISHED ================================')

pl.show()
