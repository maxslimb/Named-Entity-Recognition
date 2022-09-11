# Named Entity Recognition Using RNN

Named entity recognition is an important task in NLP. High performance approaches have been dominated by applying CRF, SVM, or perceptron models to hand-crafted features. A well-studied solution for a neural network to process variable length input and have long term memory is the recurrent neural network (RNN). RNNs have shown great success in diverse NLP tasks such as speech recognition, machine translation, and language modeling . 

# Implementation Steps
# 1. Load the dataset

We will have to use encoding = ‘unicode_escape’ while loading the data. This function takes a parameter to toggle the wrapping quotes’ addition and escaping that quote’s quote in a string. The sentences in the dataset are tokenized in the column “Word”. The column “sentence #” displays the sentence number once and then prints NaN until the next sentence begins. The ‘Tag’ column will be our label (y).
# 2. Extract mappings required for the neural network
To train a neural network, we will use two mappings as given below.
{token} to {token id}: address the row in embeddings matrix for the current token.
{tag} to {tag id}: one-hot ground truth probability distribution vectors for computing the loss at the network’s output.
The function adds two new index columns for our X (Word_idx) and y (Tag_idx) variables. Next, it will collect tokens into arrays in the respective sequence to make the recurrent neural network’s best use.
 
# 3. Transform columns to extract sequential data
To transform columns into sequential arrays, we will
Fill NaN in the ‘sentence #’ column using method ffill in fillna.
After that, run groupby on the sentence column to get a list of tokens and tags.
# 4. Split the dataset into train, test after padding
Padding: The LSTM layers accept sequences of the same length only. Therefore, every sentence represented as integers (‘Word_idx’) must be padded to have the same length. We will work with the max length of the longest sequence and pad the shorter sequences to achieve this. We will also be converting the y variable as a one-hot encoded vector using the to_categorical function in Keras. 
# 5. Build the model architecture
Neural network models work with graphical structure. Therefore we will first need to design the architecture and set input and out dimensions for every layer. RNNs are capable of handling different input and output combinations. We will use many to many architectures for this task. Our task is to output tag (y) for a token (X) ingested at each time step. In this architecture, we are primarily working with three layers (embedding, bi-lstm, lstm layers) and the 4th layer, which is TimeDistributed Dense layer, to output the result. 
# 6. Fit the model
We will fit the model with a for loop to save and visualize the loss at every epoch.
# Applications
# Classifying content for news providers:
News and publishing houses generate large amounts of online content on a daily basis and managing them correctly is very important to get the most use of each article. Named Entity Recognition can automatically scan entire articles and reveal which are the major people, organizations, and places discussed in them. Knowing the relevant tags for each article helps in automatically categorizing the articles in defined hierarchies and enables smooth content discovery. An example of how this work can be seen in the example below.The Named Entity Recognition API has successfully identified all the relevant tags for the article and this can be used for categorization.
# Efficient Search Algorithms:

Let’s suppose you are designing an internal search algorithm for an online publisher that has millions of articles. If for every search query the algorithm ends up searching all the words in millions of articles, the process will take a lot of time. Instead, if Named Entity Recognition can be run once on all the articles and the relevant entities (tags) associated with each of those articles are stored separately, this could speed up the search process considerably. With this approach, a search term will be matched with only the small list of entities discussed in each article leading to faster search execution.

# Customer Support:
There are a number of ways to make the process of customer feedback handling smooth and Named Entity Recognition could be one of them. Let’s take an example to understand the process. If you are handling the customer support department of an electronic store with multiple branches worldwide, you go through a number mentioned in your customers’ feedback.Similarly, there can be other feedback tweets and you can categorize them all on the basis of their locations and the products mentioned. You can create a database of the feedback categorized into different departments and run analytics to assess the power of each of these departments.
