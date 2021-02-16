import numpy as np
import pandas as pd

from constants import DATA

from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

tfv = TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word',
    token_pattern=r'\w{1,}', ngram_range=(1,3), use_idf=1, smooth_idf=1, sublinear_tf=1)

ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 1))

# Multi-class log-loss as evaluation metric
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

def extract_tf_idf(*args, test=False):
    if not test:
        # extract training values
        xtrain = args[0]
        xvalid = args[1]
        tfv.fit(list(xtrain) + list(xvalid))
        xtrain_tfv = tfv.transform(xtrain)
        xvalid_tfv = tfv.transform(xvalid)
        return xtrain_tfv, xvalid_tfv

    # extract testing values
    xtest = args[0]
    return tfv.transform(xtest)

def extract_ctv(*args, test=False):
    if not test:
        # extract training values
        xtrain = args[0]
        xvalid = args[1]
        ctv.fit(list(xtrain) + list(xvalid))
        xtrain_ctv = ctv.transform(xtrain)
        xvalid_ctv = ctv.transform(xvalid)
        return xtrain_ctv, xvalid_ctv

    # extract testing values
    xtest = args[0]
    return ctv.transform(xtest)
    
def logistic_reg(feature_type='ctv') -> None:
    # Read train data and model
    train_dataset = pd.read_pickle(DATA['balanced_train'])
    train_lb_encoder = preprocessing.LabelEncoder()
    train_y = train_lb_encoder.fit_transform(train_dataset.label.values)

    xtrain, xvalid, ytrain, yvalid = train_test_split(train_dataset.preprocessed_tweets.values,
                        train_y, stratify=train_y, random_state=42, test_size=.1, shuffle=True)
    
    if feature_type == 'ctv':
        xtrain_featured, xvalid_featured = extract_ctv(xtrain, xvalid)
    else:
        xtrain_featured, xvalid_featured = extract_tf_idf(xtrain, xvalid)

    model = RandomForestClassifier(n_jobs=-1, random_state=42)
    model.fit(xtrain_featured, ytrain)

    predictions = model.predict_proba(xvalid_featured)
    fscore_pred = model.predict(xvalid_featured)

    print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    print ("f score: %0.3f" % f1_score(yvalid, fscore_pred))

    # Read test data and evaluate
    test_dataset = pd.read_pickle(DATA['test'])
    test_lb_encoder = preprocessing.LabelEncoder()
    test_y = test_lb_encoder.fit_transform(test_dataset.label.values)

    xtest = test_dataset.preprocessed_tweets.values

    if feature_type == 'ctv':
        xtest_featured = extract_ctv(xtest, test=True)
    else:
        xtest_featured = extract_tf_idf(xtest, test=True)

    score = model.predict(xtest_featured)
    print ("Test data f score: %0.3f" % f1_score(test_y, score))

if __name__ == '__main__':
    logistic_reg('ctv')