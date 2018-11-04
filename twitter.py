import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from testing import removePunctuation, findC

def main():

    notFinished = True

    while(notFinished):

        try:
            train = input("Train file without extension: ")
            trainOrig = pd.read_csv(train + ".csv", encoding = "ISO-8859-1")
        except FileNotFoundError:
            print("Train not found...")
        else:
            try:
                test = input("Test file without extension: ")
                testOrig = pd.read_csv(test + ".csv", encoding = "ISO-8859-1")
            except FileNotFoundError:
                print("Test not found...")
            train = trainOrig.drop(["ItemID", "Sentiment"], axis=1).values.flatten()
            test = testOrig.drop(["ItemID"], axis=1).values.flatten()

            input("Press return to remove punctuation.")
            removePunctuation(train)
            removePunctuation(test)

            target = list(trainOrig.drop(["ItemID", "SentimentText"], axis=1).values.flatten())

            highestC, X, X_test = findC(train, test, target)

            lr = LogisticRegression(C=highestC)
            lr.fit(X, target)
            predictedSentiments = lr.predict(X_test)

            testPredicted = pd.DataFrame(predictedSentiments, columns = ['Sentiment'])
            testOrig.insert(loc=1, column="Sentiment", value=testPredicted)
            print(testOrig)
            notFinished = False


if __name__ == '__main__':
    main()


#print(list(train.drop(["ItemId", "Sentiment"], axis = 1).values.T.flatten()))
