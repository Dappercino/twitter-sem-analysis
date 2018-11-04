import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def removePunctuation(listofstrings):
    for tweetNo in range(len(listofstrings)):
        listofstrings[tweetNo] = re.sub(r"[^\w\s]", '', listofstrings[tweetNo])
        listofstrings[tweetNo] = re.sub(r"[\_]", '', listofstrings[tweetNo])
        listofstrings[tweetNo] = re.sub(r"\w*\d\w*", '', listofstrings[tweetNo])


def main():

    notFinished = True

    while(notFinished):

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

            input("Press return to remove punctuation.")
            removePunctuation(train)
            removePunctuation(test)

            #code ahead taken from towardsdatascience.com
            cv = CountVectorizer(binary=True, encoding = "ISO-8859-1")
            cv.fit(train)
            X = cv.transform(train)
            X_test = cv.transform(test)

            target = list(trainOrig.drop(["ItemID", "SentimentText"], axis=1).values.flatten())
            target.extend([0 for i in range(len(test)-len(train))])
            assert(len(target) == len(test))

            #code ahead again taken from towardsdatascience.com
            X_train, X_val, y_train, y_val = train_test_split(
                X, target, train_size = 0.75
            )

            for c in [0.01, 0.05, 0.25, 0.5, 1]:
                
                lr = LogisticRegression(C=c)
                lr.fit(X_train, y_train)
                print ("Accuracy for C=%s: %s" 
            % (c, accuracy_score(y_val, lr.predict(X_val))))


        notFinished = False

if __name__ == '__main__':
    main()


#print(list(train.drop(["ItemId", "Sentiment"], axis = 1).values.T.flatten()))
