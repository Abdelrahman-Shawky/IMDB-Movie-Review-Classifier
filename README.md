# Movies Reviews Classification Using BERT
Training and test data obtained from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
## Problem Statement
IMDB is the most globally famous movie reviews website where you can publish a review for
any film you watched. **Classifying** the positive reviews and the negative ones can be useful for
several purposes such as giving an overall rating for the film or making statistical analysis about
the preferences of people from different countries, age levels, etc... So IMDB dataset is released
which composed of 50k reviews labeled as positive or negative to enable training movie reviews
classifiers. Moreover, **NLP tasks** are currently solved based on pre-trained language models such
as **BERT**. These models provide a deep understanding of both semantic and contextual aspects
of language words, sentences, or even large paragraphs due to their training on huge corpus for
a very long time.
## Objectives
1. Applying state-of-the-art language model BERT to solve **NLP classification problem**.
2. Tackle a real-life AI problem.
## Requirements
### Data Split
Split the dataset randomly so that the training set would form 70% of the dataset, the
validation set would form 10% and the testing set would form 20% of it. All the splits should be kept balanced.
### Text Pre-processing
Text pre-processing is essential for NLP tasks. So, we will apply the following steps on
our data before used for classification:
- Remove punctuation.
- Remove stop words.
- Lowercase all characters.
- Lemmatization of words.
### Classification using BERT
We need to build a classifier model based on BERT. We can use transformers library
supplied by hugging face to get a pre-trained and ready version of BERT model. It will
also help us to tokenize the input sentence in the BERT required form and to pad the
short sentences or trim the long ones. We will use the CLS token embedding outputs of
BERT as input to the hidden dense classification layers we need to add after BERT. This
embedding is of size 768.
We need to add 4 hidden layers of 512, 256, 128, 64 units respectively before the output
layer. We will use binary cross entropy loss and adam optimizer.
### Validation and Hyperparameter Tuning
Use the validation split to evaluate the model performance after each training epoch then
save the model checkpoint to choose the one with the best performance as the final model.
We can use dropout between dense layers to avoid overfitting if it arises.
Also, we need to tune the learning rate hyperparameter of Adam optimizer using the
performance on the validation set.
### Checking Pre-processing Importance
BERT is assumed to capture the semantic and contextual aspects of the language. So,
sometimes it is better to input the text to it without pre-processing. To check the preprocessing
importance on our task we will train the model twice, once using the preprocessed
version of data and the other using the original version then test both models
using the testing set and compare between the results.
Note that we need to repeat the validation and hyperparameter tuning steps in both
cases. Also, note that the model trained on pre-processed data must be validated and
tested using pre-processed data and vice versa.
### Weight Pruning
In **weight pruning**, we prune away k% of weights.<br>
For this, I have used the percentile function in NumPy to get the kth smallest value in the weight. 
For all values in weight smaller than k, their weight will be set to zero. Also, we have to make sure that we 
don't affect the original model. For that purpose, I have used the copy module's deepcopy function to copy the model weights.
We don't have to prune the weights for the last output layer. That is why I have used i<length-2 because the length-2 parameter corresponds to the weight for the last layer.
## Conclusion
- Results indicate that unprocessed data performed better than processed data
- This indicates that there is no need for preprocessing and cleaning the input data in order to save computational power
- On the other hand, cleaning the data might worsen BERT performance
- This is due to BERT self-attention, where it uses positional coding to interpret the position of the occurrence of a particular word thus cleaning the data could result in disturbance of such positional coding
- This indicates that stopwords which were cleaned are actually valuable for BERT’s processing
- BERT actually captures a lot more semantic information based on the surrounding of the words, not just the word itself
- Without cleaning the data, this allows the model to further train on sentences in their actual context, thus getting a better model rather than just a random combination of words
- With data cleaning, we are losing the advantage of the internal dynamic embedding that BERT uses along with its positional encoding
- This is in my opinion one of BERT’s strongest points that should be utilized efficiently
- Minimal processing could be applied guaranteeing sentence context
  - Process
    - Removing HTML words
    - Removing punctuation
    - Removing symbols
  - Don’t Process
    - Lemmatization
    - De-Capitalization
    - Removing high-frequency words
