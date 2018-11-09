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

import sys, getopt

warnings.filterwarnings("ignore")

def main(argv):
    trainfile = "train.csv"
    testfile = "test.csv"
    outfile = "out.csv"
    usage = "twitter.py --train <train csv file> --test <test csv file> --output <output file name> -v (for verbose)"
    verbose = 0
    try:
        options, args = getopt.getopt(argv, "hv", ["train=", "test=", "output="])
    except getopt.GetoptError as err:
        print(usage)
        print(str(err))
        sys.exit(2)
    for opt, arg in options:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt == "--train":
            trainfile = arg
        elif opt == "--test":
            testfile = arg
        elif opt == "--out":
            outfile = arg
        elif opt == "-v":
            verbose = 1
    
    print("Train file name is:", trainfile)
    print("Test file name is:", testfile)
    print("Output file name is:", outfile)

    try:
        trainOrig = pd.read_csv(trainfile, encoding = "ISO-8859-1")
    except FileNotFoundError:
        print("Train file not found...")
    try:
        testOrig = pd.read_csv(testfile, encoding = "ISO-8859-1")
    except FileNotFoundError:
        print("Test file not found...")

    train = trainOrig.drop(["ItemID", "Sentiment"], axis=1).values.flatten()
    test = testOrig.drop(["ItemID"], axis=1).values.flatten()
    if (verbose == 1) :
        input("Press return to remove punctuation.")
    removePunctuation(train)
    removePunctuation(test)

    target = list(trainOrig.drop(["ItemID", "SentimentText"], axis=1).values.flatten())

    highestC, X, X_test = findC(train, test, target, printVals = verbose)

    lr = LogisticRegression(C=highestC)
    lr.fit(X, target)
    predictedSentiments = lr.predict(X_test)

    testPredicted = pd.DataFrame(predictedSentiments, columns = ['Sentiment'])
    testOrig.insert(loc=1, column="Sentiment", value=testPredicted)
    testOrig.to_csv(outfile, encoding = "ISO-8859-1", index=False)


if __name__ == '__main__':
    main(sys.argv[1:])