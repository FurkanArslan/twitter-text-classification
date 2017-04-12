from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from TextCleaner import TextCleaner

class TweetClassification():
    """
    this class does classification to the tweets.
    """

    def __init__(self, classifier=LinearSVC()):
        vectorizer=TfidfVectorizer(ngram_range=(1, 6), preprocessor=TextCleaner().cleanTweetText, sublinear_tf=True,
                                      smooth_idf=True, stop_words='english', min_df=5,
                                      max_df=0.8, lowercase=True, analyzer='char', norm='l2')
        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier),
        ])

    def fit(self, X, y):
        self.pipeline.fit(X,y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X,y)
