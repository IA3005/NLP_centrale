import pandas as pd 
import numpy as np

import spacy
nlp = spacy.load('en')

import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


path_train = 'data/traindata.csv'
path_test = 'data/devdata.csv'


class Classifier:
    """The Classifier"""

    # A useful function to clean and lemmatize the data
    def clean_lemmatize(dataset):
        '''As input : the column to be lemmatized.
        This function gives as output a list of strings, 
        corresponding to the lemmatized words.'''
        clean_data = []
        for row in dataset:
            nlp_row = nlp(row)
            tokens = [word.lemma_ for word in nlp_row if (not word.is_stop) & (word.is_alpha)]
            sentence = ''
            for word in tokens:
                sentence += word + ' '
            clean_data.append(sentence)
        return clean_data

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""

        # Import the train data
        data_train = pd.read_csv(trainfile, sep = "\t", names = ["polarity", "category", "word", "offsets", "sentence"])
        
        # We clean and lemmatize the sentences and the target words (to be coherent)
        Classifier.clean_lemmatize = staticmethod(Classifier.clean_lemmatize)
        data_train.clean_sentence = Classifier.clean_lemmatize(data_train.sentence)
        data_train.clean_word = Classifier.clean_lemmatize(data_train.word)
    
        # We create the BOW vectors
        self.restaurant_vect = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)
        reviews_counts = self.restaurant_vect.fit_transform(data_train.clean_sentence)
    
        # We transform the BOW vector with the tfidf scores
        self.tfidf_transformer = TfidfTransformer()
        reviews_tfidf = self.tfidf_transformer.fit_transform(reviews_counts)
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(reviews_tfidf, data_train.polarity, 
                                                            test_size = 0.20, random_state = 5)
        # Train a Multimoda Naive Bayes classifier
        self.clf = MultinomialNB().fit(X_train, y_train)
        
        # Predicting the Test set results, find accuracy
        y_pred = self.clf.predict(X_test)
        print("Accuracy :")
        print(accuracy_score(y_test, y_pred))
        
        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
        print("Confusion Matrix : (Positive / Neutral / Negative)")
        print(cm)

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        
        # Import the test data
        data_test = pd.read_csv(datafile, sep = "\t", names = ["polarity", "category", "word", "offsets", "sentence"])
        
        # Clean the test dataset the same way as for the train dataset
        Classifier.clean_lemmatize = staticmethod(Classifier.clean_lemmatize)
        data_test.clean_sentence = Classifier.clean_lemmatize(data_test.sentence) 
        data_test.clean_word = Classifier.clean_lemmatize(data_test.word) 
        
        # Apply the same transformations 
        reviews_new_counts = self.restaurant_vect.transform(data_test.clean_sentence)
        reviews_new_tfidf = self.tfidf_transformer.transform(reviews_new_counts)
        
        # have classifier make a prediction
        self.pred = self.clf.predict(reviews_new_tfidf)
        print(accuracy_score(data_test.polarity, self.pred))
        
        # Making the Confusion Matrix
        cm2 = confusion_matrix(data_test.polarity, self.pred, labels=["positive", "neutral", "negative"])
        
        return self.pred
        





