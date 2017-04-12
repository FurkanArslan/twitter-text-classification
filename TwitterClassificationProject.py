from sklearn.model_selection import KFold
import pandas as pd
from TweetClassification import TweetClassification


class TwitterClassificationProject:
    """
    this class tests classification model.
    """

    def __init__(self, path):
        self.readTwitData(path)

    def testClassificationQuality(self):
        score = 0
        kfold = KFold(n_splits=10, shuffle=True, random_state=0)

        tweetClassification = TweetClassification()

        for ind_train, ind_test in kfold.split(self.tweets):
            dataTest = self.tweets[ind_test]
            dataTrain = self.tweets[ind_train]
            targetTest = self.target[ind_test]
            targetTrain = self.target[ind_train]

            tweetClassification.fit(dataTrain, targetTrain)

            score += tweetClassification.score(dataTest, targetTest)

        return score / 10

    def readTwitData(self, path):
        readedData = pd.read_csv(path, skipinitialspace=True)

        redundentData = readedData[readedData.Sentiment != 'irrelevant']

        self.target = redundentData.Sentiment.values
        self.tweets = redundentData.TweetText.values
