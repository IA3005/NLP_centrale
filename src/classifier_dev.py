import pandas as pd 

import spacy
nlp = spacy.load('en')

import nltk
from nltk.tokenize import word_tokenize        
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB # test_acc = 76.06
from sklearn.naive_bayes import MultinomialNB # test_acc = 77.93
from sklearn.svm import LinearSVC # test_acc = 77.13
from sklearn.svm import SVC # test_acc = 70.21


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

path_train = 'data/traindata.csv'
path_test = 'data/devdata.csv'

class Classifier:
    """The Classifier"""

    # A useful function to clean and lemmatize the data
    def create_sentence(dataset):
        '''As input : the column to be lemmatized.
        This function gives as output a list of strings, 
        corresponding to the lemmatized words.'''
        clean_data = []
        for row in dataset:
            sentence = ''
            for word in row:
                sentence += word + ' '
            clean_data.append(sentence)
        return clean_data

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""

        # We load the data and lower the text
        data_train = pd.read_csv(trainfile, sep = "\t", names = ["polarity", "category", "word", "offsets", "sentence"])
        data_train['sentence_l'] = data_train['sentence'].apply(str.lower)
        data_train['word'] = data_train['word'].apply(str.lower)
        
        # We try to keep all the no/nor/not words as this changes radically the sentiment analysis
        data_train['sentence_l'] = data_train["sentence_l"].apply(lambda sentence: sentence.replace("can\'t", "can not"))
        data_train['sentence_l'] = data_train["sentence_l"].apply(lambda sentence: sentence.replace("n\'t", " not"))
        self.stopwords = stopwords.words("english")
        self.stopwords.remove('nor')
        self.stopwords.remove('no')
        self.stopwords.remove('not')
        
        # We clean the train data and stem the words
        self.stemmer = nltk.porter.PorterStemmer()
        clean_sentences = []
        for row in data_train['sentence_l']:
            tokens = word_tokenize(row)
            #tokens = [word for word in tokens if word.isalpha()]
            tokens = [w for w in tokens if not w in self.stopwords] 
            tokens = [self.stemmer.stem(word) for word in tokens]
            clean_sentences.append(tokens)
        data_train['stems'] = clean_sentences
        
        # We also stem the target words to be coherent with the stemmed words in the sentences
        data_train['word'] = [self.stemmer.stem(word) for word in data_train['word']]
    
        # We recreate the sentences with the selected and cleaned words
        Classifier.create_sentence = staticmethod(Classifier.create_sentence)
        data_train.clean_sentence = Classifier.create_sentence(data_train.stems)
        
        # We create a BOW vector
        self.restaurant_vect = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)
        reviews_counts = self.restaurant_vect.fit_transform(data_train.clean_sentence)
    
        # We transform the BOW vector with the tfidf scores
        self.tfidf_transformer = TfidfTransformer()
        reviews_tfidf = self.tfidf_transformer.fit_transform(reviews_counts)
        
        # Split data into training and test sets
        test_size = 1
        X_train, X_test, y_train, y_test = train_test_split(reviews_tfidf, data_train.polarity, 
                                                            test_size = test_size/100, random_state = 42)
        # Train a a Linear Support Vector Classifier
        self.clf = MultinomialNB().fit(X_train, y_train)
        
        # Predicting the Test set results, find accuracy (for the test part of the training dataset)
        y_pred = self.clf.predict(X_test)
        print("#####")
        print("Accuracy on the train set (%d %% of the whole dataset) :" % test_size)
        print(accuracy_score(y_test, y_pred))
        print("#####")
        
        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
        print("Confusion Matrix for the train set :")
        print("(Positive / Neutral / Negative)")
        print(cm)
        print("#####")

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
 
        # We load the test data and lower the text
        data_test = pd.read_csv(datafile, sep = "\t", names = ["polarity", "category", "word", "offsets", "sentence"])
        data_test['sentence_l'] = data_test['sentence'].apply(str.lower)
        data_test['word'] = data_test['word'].apply(str.lower)
        
        # We try to keep all the no/nor/not words as this changes radically the sentiment analysis
        data_test['sentence_l'] = data_test["sentence_l"].apply(lambda sentence: sentence.replace("can\'t", "can not"))
        data_test['sentence_l'] = data_test["sentence_l"].apply(lambda sentence: sentence.replace("n\'t", " not"))
        
        # We clean the data and stem the words
        clean_sentences = []
        for row in data_test['sentence_l']:
            tokens = word_tokenize(row)
            tokens = [word for word in tokens if word.isalpha()]
            tokens = [w for w in tokens if not w in self.stopwords] 
            tokens = [self.stemmer.stem(word) for word in tokens]
            clean_sentences.append(tokens)
        data_test['stems'] = clean_sentences
        
        # We also stem the target words to be coherent with the stemmed words in the sentences
        data_test['word'] = [self.stemmer.stem(word) for word in data_test['word']]

        # We recreate the sentences with the selected and cleaned words
        Classifier.create_sentence = staticmethod(Classifier.create_sentence)
        data_test.clean_sentence = Classifier.create_sentence(data_test.stems)
        
        # We create a BOW vector
        reviews_new_counts = self.restaurant_vect.transform(data_test.clean_sentence)
        
        # We transform the BOW vector with the tfidf scores
        reviews_new_tfidf = self.tfidf_transformer.transform(reviews_new_counts)
        
        # We make a prediction with the classifier
        self.pred = self.clf.predict(reviews_new_tfidf)
        print("#####")
        print("Accuracy on the test set:")
        print(accuracy_score(data_test.polarity, self.pred))
        print("#####")
        
        # Making the Confusion Matrix
        cm2 = confusion_matrix(data_test.polarity, self.pred, labels=["positive", "neutral", "negative"])
        print("Confusion Matrix for the test set :")
        print("(Positive / Neutral / Negative)")
        print(cm2)
        print("#####")
        
        return self.pred
        





