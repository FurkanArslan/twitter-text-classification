import re
import string

class TextCleaner():

    """
    this class cleans up a text.
    """

    def __init__(self):
        self.noisePattern = re.compile("|".join(["http\S+", "\@", "\#", '"']))
        self.remove_ellipsis_re = re.compile(r'\.\.\.')
        self.punctuationPattern = re.compile('[%s]' % re.escape(string.punctuation))
        self.pricePattern = re.compile(r"\d+\.\d\d")
        self.numberPattern = re.compile(r"\d+")

    def cleanTweetText(self, text):
        """
        this function will take a text and try to cleanup it.

        Parameters
        ----------
        text: str
              the text will be cleaned
        Returns
        -------
        text_cleaned: str
                         cleaned up version of the raw text
        """

        text_cleaned = text.lower()
        text_cleaned = re.sub(self.noisePattern, '', text_cleaned)
        text_cleaned = re.sub(self.pricePattern, 'PRICE', text_cleaned)
        text_cleaned = re.sub(self.remove_ellipsis_re, '', text_cleaned)
        text_cleaned = re.sub(self.punctuationPattern, '', text_cleaned)
        text_cleaned = re.sub(self.numberPattern, 'NUM', text_cleaned)

        return text_cleaned
