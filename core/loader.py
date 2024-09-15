import re
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataLoader:
    def __init__(self, data_path, target_column="label", text_column="text"):
        """
        :param data_path: path to the data file
        """
        self.data_path = data_path
        self.data = self.load_data()
        self.target_column = target_column
        self.text_column = text_column

    def load_data(self):
        """
        Load the data from the data_path
        :return: pandas dataframe
        """
        return pd.read_csv(self.data_path, index_col=False)
    
    # perform some basic analysis on the data and return the results in a dictionary
    def basic_analysis(self):
        """
        Perform some basic analysis on the data
        :return: dictionary
        """
        return {
            "shape": self.data.shape,
            "missing": self.check_missing(),
            "duplicates": self.data.duplicated().sum(),
            "balance": self.check_balance(),
            "class_distribution": self.class_distribution()
        }
    
    # check if the data has any missing values
    def check_missing(self):
        """
        Check if the data has any missing values
        :return: boolean
        """
        return self.data.isnull().sum().any()
    
    # drop the missing values
    def drop_missing(self):
        """
        Drop the missing values
        :return: pandas dataframe
        """
        return self.data.dropna()
    
    # fill the missing values with the mean of the column
    def fill_missing(self):
        """
        Fill the missing values with the mean of the column
        :return: pandas dataframe
        """
        return self.data.fillna(self.data.mean())
    
    # drop the duplicates
    def drop_duplicates(self):
        """
        Drop the duplicates
        :return: pandas dataframe
        """
        return self.data.drop_duplicates()
    
    # check if the labels are balanced or not and return 
    def check_balance(self):
        """
        Check if the labels are balanced or not
        :param target: target column
        :return: boolean
        """
        return self.data[self.target_column].value_counts().min() / self.data[self.target_column].value_counts().max() > 0.2
    
    # return the number of samples for each class
    def class_distribution(self):
        """
        Return the number of samples for each class
        :param target: target column
        :return: dictionary
        """
        return self.data[self.target_column].value_counts().to_dict()
    
    # get the text data from text column of the dataframe, tokenize it and return the frequency of each word
    def get_word_frequency(self):
        """
        Get the word frequency from the text column
        :param text_column: text column
        :return: dictionary
        """
        text = " ".join(self.data[self.text_column].values)
        tokens = word_tokenize(text)
        return Counter(tokens)
    
    # get the text data from text column of the dataframe by the target class, tokenize it and return the frequency of each word
    def get_word_frequency_by_class(self):
        """
        Get the word frequency by class
        :param text_column: text column
        :param target_column: target column
        :return: dictionary
        """
        word_freq = {}
        for label in self.data[self.target_column].unique():
            text = " ".join(self.data[self.data[self.target_column] == label][self.text_column].values)
            tokens = word_tokenize(text)
            word_freq[label] = Counter(tokens)
        return word_freq
    
    # preprocess the text data by removing stopwords and special characters
    def preprocess_text(self):
        """
        Preprocess the text data
        :param text_column: text column
        :return: pandas dataframe
        """
        stop_words = set(stopwords.words('english'))
        self.data[self.text_column] = self.data[self.text_column].apply(lambda x: " ".join([word for word in word_tokenize(x.lower()) if word.isalnum() and word not in stop_words]))
        return self.data

    # remove all the top n most common words from the text data
    def remove_common_words(self, top_n=10):
        """
        Remove the top N most common words from the text data
        :param text_column: text column
        :param top_n: number of top words to remove
        :return: pandas dataframe
        """
        word_freq = self.get_word_frequency()
        common_words = [word for word, _ in word_freq.most_common(top_n)]
        self.data[self.text_column] = self.data[self.text_column].apply(lambda x: " ".join([word for word in word_tokenize(x) if word not in common_words]))
        return self.data