from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from math import sqrt

def load_data(split):
    with open('CR.' + split) as f:
        data = []
        for line in f.readlines():
            if split != 'test':
                text, sentiment = line.split('\t')
                data.append([text.strip(), sentiment[0] == '1'])
            else:
                text = line
                data.append([text.strip(), False])
        return data
    
def scaleX(X):
    """
    unit length scaling of each example
    """
    Xscaled=np.copy(X);
    for i in range(Xscaled.shape[0]):
        sqsum = sqrt(np.sum(Xscaled[i]**2))
        if not sqsum == 0:
            Xscaled[i] = Xscaled[i]/sqsum
    return Xscaled
    
def toXy(data, key_to_ix):
    X = np.vstack([textToFeature(txt, key_to_ix) for txt, _ in data])
    X = scaleX(X)
    y = np.array([int(sentiment) * 2 - 1 for _, sentiment in data])
    return X, y

def textToFeature(text, key_to_ix):
    words = text.split()
    feat = np.zeros((len(key_to_ix)))
    akey = next (iter (key_to_ix.keys()))
    if isinstance(akey, tuple): # bigram
        for i in range(len(words) - 1):
            if (words[i], words[i + 1]) in key_to_ix:
                feat[key_to_ix[(words[i], words[i + 1])]] = 1
                #feat[key_to_ix[(words[i], words[i + 1])]] += 1
        
    else:
        for word in words:
            if word in key_to_ix:
                feat[key_to_ix[word]] = 1
                #feat[key_to_ix[word]] += 1 # if we want actual counts, not just present/absent
    
    return feat
    
def getVocab(data, use_bigram,mincount):
    """
    construct vocabulary from data
    only include terms that appear at least mincount times
    """
    word_count = {}
    if use_bigram:
        for text, _ in data:
            words = text.split()
            for i in range(len(words) - 1):
                word_count[(words[i], words[i + 1])] = word_count.get((words[i], words[i + 1]), 0) + 1
    else:
        for text, _ in data:
            for word in text.split():
                word_count[word] = word_count.get(word, 0) + 1
                
    for word in list(word_count):
        if word_count[word] < mincount:
            del word_count[word]
    print("Feature size: ", len(word_count))
    # print(word_count.keys())
    keys = word_count.keys()
    key_to_ix = {key:ix for ix, key in enumerate(keys)}
    
    return keys, key_to_ix

def preprocess(use_bigram = False,mincount=5):
    """
    Preprocessing:
    Load text and label data.
    Build the vocabulary, and transform text into binary features.
    """
    train_data, val_data, test_data = load_data('train'), load_data('dev'), load_data('test')
    data = train_data + val_data + test_data
    keys, key_to_ix = getVocab(data, use_bigram,mincount)
    X = {}
    y = {}
    X['train'], y['train'] = toXy(train_data, key_to_ix)
    X['val'], y['val'] = toXy(val_data, key_to_ix)
    X['test'], y['test'] = toXy(test_data, key_to_ix)
    
    return X, y, keys

def test_linear_SVM(svm, num_samples=10, num_features=2):
    """
    test linear svm
    """
    X = np.random.random((num_samples, num_features)) * 2 - 1
    y = 2 * (X.sum(1) > 0.5) - 1.0
    
    svm.fit(X, y)
    plot(svm, X, y, 100)
    
def test_rbf_SVM(svm, num_samples=10, num_features=2):
    """
    test rbf svm
    """
    X = np.random.random((num_samples, num_features)) * 2 - 1
    y = 2 * (np.sum(X ** 2, 1) > 0.5) - 1.0
    
    svm.fit(X, y)
    plot(svm, X, y, 100)

import pylab as plt
import itertools
import matplotlib.cm as cm
    
def plot(svm, X, y, grid_size):
    x_min, x_max = X[:, 0].min() , X[:, 0].max()
    y_min, y_max = X[:, 1].min() , X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(svm.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)
    #plt.scatter(flatten(xx[:, 0]), flatten(xx[:, 1]),
    #            c=flatten(yy), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.plot()
    
def save_submission(filename, yhats):
    assert np.ndim(yhats) == 1
    id_and_prediction = np.vstack([np.arange(len(yhats)).T, yhats]).T
    np.savetxt(filename, id_and_prediction,
               fmt='%d',
               delimiter=',',
               comments='',
               header='Id,Prediction')