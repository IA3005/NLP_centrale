import pandas as pd 

import nltk
from nltk.tokenize import word_tokenize        
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB # With alpha=0.53 acc=78.72
from sklearn.svm import LinearSVC # With C=0.138 acc=79.26


class Classifier:
    """The Classifier"""

    # A useful function 
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
        data_train['sentence_l'] = data_train.sentence.apply(str.lower)
        data_train['word'] = data_train.word.apply(str.lower)
        
        # We try to keep all the no/nor/not words as this changes radically the sentiment analysis
        data_train['sentence_l'] = data_train.sentence_l.apply(lambda sentence: sentence.replace("can\'t", "can not"))
        data_train['sentence_l'] = data_train.sentence_l.apply(lambda sentence: sentence.replace("n\'t", " not"))
        self.stopwords = stopwords.words("english")
        self.stopwords.remove('nor')
        self.stopwords.remove('no')
        self.stopwords.remove('not')
        
        # We clean the train data and stem the words
        self.stemmer = nltk.porter.PorterStemmer()
        clean_sentences = []
        for row in data_train['sentence_l']:
            tokens = word_tokenize(row)
            tokens = [word for word in tokens if word.isalpha()]
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
        
        polarities = []
        for row in data_train['polarity']:
            if row == 'positive':
                polarities.append(1)
            if row == 'neutral':
                polarities.append(0)
            if row == 'negative':
                polarities.append(-1)
        data_train['polarity_floats'] = polarities
        
        # Split data into training and test sets
        test_size = 10
        X_train, X_test, y_train, y_test = train_test_split(reviews_tfidf, data_train.polarity_floats,
                                                            test_size = test_size/100, random_state = None)
        
        ############# CNN MODEL ##############
        
        from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
        from keras.layers import Reshape, Flatten, Dropout, Concatenate
        from keras.optimizers import Adam
        from keras.models import Model
        
        sequence_length = X_train.shape[1] # 7
        vocabulary_size = X_train.shape[0] # 1503
        embedding_dim = 256
        filter_sizes = [3,4,5]
        num_filters = 512
        drop = 0.5
        
        epochs = 10
        batch_size = 50
        
        # this returns a tensor
        inputs = Input(shape=(sequence_length,), dtype='int32')
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
        reshape = Reshape((sequence_length,embedding_dim,1))(embedding)
        
        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        
        maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
        
        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=1, activation='softmax')(dropout)
        
        # this creates a model that includes
        model = Model(inputs=inputs, outputs=output)
        
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        print("Training Model...")
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))  # starts training
        

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
 
        # We load the test data and lower the text
        data_test = pd.read_csv(datafile, sep = "\t", names = ["polarity", "category", "word", "offsets", "sentence"])
        data_test['sentence_l'] = data_test.sentence.apply(str.lower)
        data_test['word'] = data_test.word.apply(str.lower)
        
        # We try to keep all the no/nor/not words as this changes radically the sentiment analysis
        data_test['sentence_l'] = data_test.sentence_l.apply(lambda sentence: sentence.replace("can\'t", "can not"))
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
        self.pred = self.model.predict(reviews_new_tfidf)
        
        return self.pred
        





