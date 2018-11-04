import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def findC(trainList, testList, targetList, printVals=False):

    #code ahead taken from towardsdatascience.com
    cv = CountVectorizer(binary=True, encoding="ISO-8859-1")
    cv.fit(trainList)
    X = cv.transform(trainList)
    X_test = cv.transform(testList)

    #create sample for training
    X_train, X_val, y_train, y_val = train_test_split(
        X, targetList, train_size = 0.75
    )

    highestC = 0
    highestAcc = 0

    for c in [0.1*i for i in range(1, 11)]:
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        accuracy = accuracy_score(y_val, lr.predict(X_val))
        if (accuracy > highestAcc):
            highestC = c
            highestAcc = accuracy
        if(printVals):
            print ("Accuracy for C=%s: %s" 
                % (c, accuracy))
    return highestC, X, X_test

def removePunctuation(listofstrings, removeUnderscores=False, removeNumWords=False):
    for tweetNo in range(len(listofstrings)):
        listofstrings[tweetNo] = re.sub(r"[^\w\s]", '', listofstrings[tweetNo])
        if (removeUnderscores):
            listofstrings[tweetNo] = re.sub(r"[\_]", '', listofstrings[tweetNo])
        if (removeNumWords):
            listofstrings[tweetNo] = re.sub(r"\w*\d\w*", '', listofstrings[tweetNo])

def main():
    try:
        train = input("Train file without extension: ")
        train = pd.read_csv(train + ".csv", encoding = "ISO-8859-1")
    except FileNotFoundError:
        print("Train not found...")
    else:
        try:
            test = input("Test file without extension: ")
            test = pd.read_csv(test + ".csv", encoding = "ISO-8859-1")
        except FileNotFoundError:
            print("Test not found...")
        trainOrig = train
        testOrig = test
        train = train.drop(["ItemID", "Sentiment"], axis=1).values.flatten()
        test = test.drop(["ItemID"], axis=1).values.flatten()
        test = test[:len(train)]

        input("Press return to remove punctuation.")
        removePunctuation(train)
        removePunctuation(test)

        target = list(trainOrig.drop(["ItemID", "SentimentText"], axis=1).values.flatten())

        highestC, X, X_test = findC(train, test, target)

        final_model = LogisticRegression(C=highestC)
        final_model.fit(X, target)
        print ("Final Accuracy: %s" 
            % accuracy_score(target, final_model.predict(X_test)))

if __name__ == '__main__':
    main()