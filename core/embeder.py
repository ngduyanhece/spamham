import re
from collections import Counter

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


class DataEmbeder:
    def __init__(self, text_data):
        self.text_data = text_data

    def tfidf(self):
        """
        Perform tfidf embedding on the text data
        :param data: pandas dataframe
        :param text_column: name of the text column
        :return: numpy array
        """
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(self.text_data), vectorizer
    
    def cluster_tfidf(self, num_clusters=2):
        """
        Perform clustering on the tfidf data
        :param data: pandas dataframe
        :param text_column: name of the text column
        :param num_clusters: number of clusters
        :return: numpy array, cluster centers
        """
        tfidf_data, vectorizer = self.tfidf()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(tfidf_data)
        return kmeans.labels_, tfidf_data, kmeans.cluster_centers_, vectorizer

    def get_top_words(self, cluster_centers, vectorizer, top_n=5):
        """
        Get the top N words for each cluster
        :param cluster_centers: cluster centers
        :param vectorizer: fitted TfidfVectorizer
        :param top_n: number of top words to extract
        :return: list of top words for each cluster
        """
        top_words = []
        terms = vectorizer.get_feature_names_out()
        for i in range(cluster_centers.shape[0]):
            center = cluster_centers[i]
            top_indices = center.argsort()[-top_n:][::-1]
            top_words.append([terms[index] for index in top_indices])
        return top_words

    def display_tfidf_clusters_tsne(self, num_clusters=2):
        """
        Display the clusters with a graph using t-SNE
        :param data: pandas dataframe
        :param text_column: name of the text column
        :param num_clusters: number of clusters
        """
        labels, tfidf_data, cluster_centers, vectorizer = self.cluster_tfidf(num_clusters)
        
        # Reduce dimensionality to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(tfidf_data.toarray())
        
        # Get the top words for each cluster
        top_words = self.get_top_words(cluster_centers, vectorizer)
        cluster_labels = ['#'.join(words) for words in top_words]
        
        # Create a DataFrame for plotting
        plot_data = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'label': [cluster_labels[label] for label in labels]
        })
        
        # Plot the clusters
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='x', y='y', hue='label', data=plot_data, palette='viridis')
        plt.title('TF-IDF Clusters with t-SNE')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Cluster')
        plt.show()

    def get_top_words_tfidf(self, top_n=10):
        """
        Get the top N words based on tfidf values
        :param text_data: text data
        :param top_n: number of top words to extract
        :return: list of top words and their tfidf values
        """
        tfidf_data, vectorizer = self.tfidf()
        terms = vectorizer.get_feature_names_out()
        tfidf_values = tfidf_data.toarray().sum(axis=0)
        top_indices = tfidf_values.argsort()[-top_n:][::-1]
        top_words = [(terms[index], tfidf_values[index]) for index in top_indices]
        return top_words
    
    # embed the text data using glove embeddings
    def glove(self):
        """
        Perform GloVe embedding on the text data
        :return: numpy array
        """
        glove = api.load('glove-wiki-gigaword-100')
        embeddings = []
        for text in self.text_data:
            words = word_tokenize(text)
            word_vectors = [glove[word] for word in words if word in glove]
            if word_vectors:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(glove.vector_size))
        
        return np.array(embeddings)
    
    def cluster_glove(self, num_clusters=2):
        """
        Perform clustering on the GloVe data
        :param num_clusters: number of clusters
        :return: numpy array, cluster centers
        """
        glove_data = self.glove()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(glove_data)
        return kmeans.labels_, glove_data, kmeans.cluster_centers_
    
    def display_glove_clusters_tsne(self, num_clusters=2):
        """
        Display the clusters with a graph using t-SNE
        :param num_clusters: number of clusters
        """
        labels, glove_data, cluster_centers = self.cluster_glove(num_clusters)
        # get the top words back from tfidf
        # Reduce dimensionality to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(glove_data)
        
        # Create a DataFrame for plotting with the cluster labels
        plot_data = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'label': labels
        })

        # Plot the clusters
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='x', y='y', hue='label', data=plot_data, palette='viridis')
        plt.title('GloVe Clusters with t-SNE')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Cluster')
        plt.show()
    
    # embed one word using glove embeddings if not return zero vector
    def get_word_vector(self, word):
        """
        Get the GloVe embedding for a word
        :param word: word
        :return: numpy array
        """
        glove = api.load('glove-wiki-gigaword-100')
        return glove.get(word, np.zeros(100))