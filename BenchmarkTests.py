from sklearn.model_selection import KFold , GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier , LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re
import string

class BenchmarkClassification():
    """
    this class does classification to the tweets.
    """

    def __init__(self, path):
        self.readTwitData(path)

    def readTwitData(self, path):
        readedData = pd.read_csv(path, skipinitialspace = True)

        redundentData = readedData[readedData.Sentiment != 'irrelevant']

        self.testData = redundentData.TweetText.values
        self.testTarget = redundentData.Sentiment.values

    def remove_noise(self, text):
        noisePattern = re.compile("|".join(["http\S+", "\@", "\#", '"']))
        remove_ellipsis_re = re.compile(r'\.\.\.')
        punct_re = re.compile('[%s]' % re.escape(string.punctuation))
        price_re = re.compile(r"\d+\.\d\d")
        number_re = re.compile(r"\d+")

        t = text.lower()
        t = re.sub(noisePattern, '', t)
        t = re.sub(price_re, 'PRICE', t)
        t = re.sub(remove_ellipsis_re, '', t)
        t = re.sub(punct_re, '', t)
        t = re.sub(number_re, 'NUM', t)

        return t

    def evaluate_cross_validation(self, clf, data, target, cluster):
        score = 0
        kfold = KFold(n_splits=cluster, shuffle=True, random_state=0)

        for ind_train, ind_test in kfold.split(data):
            dataTest  = data[ind_test]
            dataTrain = data[ind_train]
            targetTest = target[ind_test]
            targetTrain = target[ind_train]

            clf.fit(dataTrain, targetTrain)

            score += clf.score(dataTest, targetTest)

        print ('-'*30)
        print ("Mean score: %0.3f" % (score/10))
        print ('-'*30)

        return score/10

    def benchmarkClassification(self):
        maxScore = 0
        bestClassification = None

        tfidfVectorizerWithChar = TfidfVectorizer(ngram_range=(1, 6), preprocessor=self.remove_noise, stop_words='english',
                                     min_df = 5, max_df=0.8, sublinear_tf = True, smooth_idf=True,
                                     lowercase=True, use_idf=True, analyzer='char', norm = 'l2')

        tfidfVectorizerWithWord = TfidfVectorizer(preprocessor=self.remove_noise, lowercase=True,
                                     ngram_range = (1,2), use_idf=True, stop_words='english' )

        clf_1 = Pipeline([
            ('vect', tfidfVectorizerWithChar),
            ('clf', MultinomialNB(alpha=0.1)),
        ])
        clf_2 = Pipeline([
            ('vect', tfidfVectorizerWithChar),
            ('clf', LinearSVC()),
        ])
        clf_3 = Pipeline([
            ('count_vectorizer',   CountVectorizer(ngram_range=(1,6), analyzer='char', min_df= 5, max_df=0.8,
                                                   preprocessor=self.remove_noise, lowercase=True,stop_words='english')),
            ('tfidf_transformer',  TfidfTransformer(use_idf=True,sublinear_tf = True, smooth_idf=True)),
            ('classifier',         LinearSVC())
        ])
        clf_4 = Pipeline([
            ('vect', tfidfVectorizerWithWord),
            ('clf', LogisticRegression(C=10.0))
        ])
        clf_5 = Pipeline([
            ('vect', tfidfVectorizerWithWord),
            ('clf', SGDClassifier(n_iter=100)),
        ])
        clf_6 =  Pipeline([
            ('vect', tfidfVectorizerWithChar),
            ('clf', BernoulliNB(alpha=0.1)),
        ])
        clf_7 =  Pipeline([
            ('vect', tfidfVectorizerWithWord),
            ('clf', RidgeClassifier(alpha = 0.1)),
        ])
        clf_8 =  Pipeline([
            ('vect', tfidfVectorizerWithChar),
            ('clf', Perceptron(n_iter=100)),
        ])
        clf_9 =  Pipeline([
            ('vect', tfidfVectorizerWithChar),
            ('clf', PassiveAggressiveClassifier(n_iter=100)),
        ])
        clf_10 =  Pipeline([
            ('vect', tfidfVectorizerWithChar),
            ('clf', KNeighborsClassifier(n_neighbors=10)),
        ])
        clf_11 =  Pipeline([
            ('vect', tfidfVectorizerWithChar),
            ('clf', RandomForestClassifier(n_estimators=100)),
        ])

        clfs = [clf_1, clf_2, clf_3,clf_4, clf_5, clf_6,clf_7,clf_8,clf_9,clf_10,clf_11]

        i=1
        for clf in clfs:
            print ('clf_' + str(i))
            i +=1
            score = self.evaluate_cross_validation(clf, self.testData, self.testTarget, 10)

            if score > maxScore:
                maxScore = score
                bestClassification = clf

        print ('Best Classification Method:\n', bestClassification)
        print ('Best Score: ' + str(maxScore))

        return bestClassification

    def findBestParameters(self, classification, parameters, cluster):
        gs = GridSearchCV(classification, parameters, verbose=2, refit=False, cv=cluster)
        gs.fit(self.testData, self.testTarget)

        print (gs.best_params_, gs.best_score_)
