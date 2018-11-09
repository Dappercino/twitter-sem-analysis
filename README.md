# twitter-sem-analysis
A non-competitive attempt at Kaggle's 'Twitter sentiment analysis' challenge.

Requires two files, train.csv and test.csv, to be in the directory. The train file should have a list of tweets and an associated emotional value. This will be used to make predictions about the tweets in test. Finally, a version of test will be created that will have the predicted sentiments associated with the tweets. The train and test files were provided by Kaggle.

The train file has three columns, ItemID (the tweet #), Sentiment (a binary number: 0 means the tweet is negative, 1 positive), and SentimentText, the tweet contents.

The test file has the same columns as train except it doesn't have Sentiment, the column which will (hopefully) be predicted using this program.

To start, just run twitter.py. It'll search its current folder for train, test csv files to use, and output the predicted tweets in out.csv. These names can be changed using the relevant arguments:

Usage: twitter.py --train <name of train.csv file> --test <name of test.csv file> --output <output file name> -v (for verbose)
