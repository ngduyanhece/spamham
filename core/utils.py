import matplotlib.pyplot as plt
import seaborn as sns


def display_graphs(analysis_result):
    """
    Display graphs for the analysis result.
    
    Parameters:
    analysis_result (dict): The dictionary returned from the basic_analysis function.
    """
    # Plot the shape of the dataset
    plt.figure(figsize=(6, 4))
    plt.title('Shape of the Dataset')
    plt.bar(['Rows', 'Columns'], analysis_result['shape'])
    plt.ylabel('Count')
    plt.show()
    
    # Plot the class distribution
    plt.figure(figsize=(6, 4))
    plt.title('Class Distribution')
    sns.barplot(x=list(analysis_result['class_distribution'].keys()), y=list(analysis_result['class_distribution'].values()))
    plt.ylabel('Count')
    plt.show()
    
    # Plot the missing values
    plt.figure(figsize=(6, 4))
    plt.title('Missing Values')
    plt.bar(['Missing'], [analysis_result['missing']])
    plt.ylabel('Count')
    plt.show()
    
    # Plot the duplicates
    plt.figure(figsize=(6, 4))
    plt.title('Duplicate Rows')
    plt.bar(['Duplicates'], [analysis_result['duplicates']])
    plt.ylabel('Count')
    plt.show()
    
    # Plot the balance
    plt.figure(figsize=(6, 4))
    plt.title('Balance of the Dataset')
    plt.bar(['Balanced'], [analysis_result['balance']])
    plt.ylabel('Count')
    plt.show()

def display_most_common_words(word_freq_tuples):
    """
    Display the most common words with a graph.
    
    Parameters:
    word_freq_tuples (list of tuples): List of tuples where each tuple contains a word and its frequency.
    """
    # Extract words and counts for plotting
    words = [word for word, _ in word_freq_tuples]
    counts = [count for _, count in word_freq_tuples]
    
    # Create a bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=counts, y=words, palette='viridis')
    plt.title('Most Common Words')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.show()


def display_most_common_words_by_class(word_freq_by_class, top_n=50):
    """
    Display the most common words by class with a graph.
    
    Parameters:
    word_freq_by_class (dict): Dictionary where keys are class labels and values are Counter objects with word frequencies.
    top_n (int): Number of top words to display for each class.
    """
    for class_label, counter in word_freq_by_class.items():
        # Extract the top N words and their counts
        most_common_words = counter.most_common(top_n)
        words = [word for word, count in most_common_words]
        counts = [count for word, count in most_common_words]
        
        # Create a bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=counts, y=words, palette='viridis')
        plt.title(f'Top {top_n} Most Common Words in {class_label.capitalize()} Class')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.show()