### Naïve Bayesian Email Classifier

Machine Learning model that classifies an email as ham or spam

This is a Naive Bayes classifier project implemented in python which basically classifies a given text/message as spam or ham. This project was for the Machine Learning graduate class.

### Description:

We are given a set of email/sms messages, m=(w1,w2,w3,.....,wn) where (w1,w2,w3, ...wn) is a set of unique words in the message we have to classified as genuine of spam.

![Bayes Theorem](https://github.com/ssrishabh96/Machine-Learning-Email-SMS-Spam-Ham-Classifier/blob/master/images/1.png "Basis of the probability")

### Dataset Used:

SMS Spam Collection Dataset:
The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged according to being ham (legitimate) or spam.

![Source: Kaggle](https://github.com/ssrishabh96/Machine-Learning-Email-SMS-Spam-Ham-Classifier/blob/master/images/2.png "Dataset Used")

Link: https://www.kaggle.com/uciml/sms-spam-collection-dataset

### Libraries Used:
![Python Std Libraries](https://github.com/ssrishabh96/Machine-Learning-Email-SMS-Spam-Ham-Classifier/blob/master/images/3.png "Python Libraries")

### Implementation:

- We first pre-process the messages. We begin by making converting the characters to lower case because case-sensitivity does never alter the meaning of any word in any sentence.
- Then, we tokenize the messages in the dataset that splits message into pieces and throwing away the punctuation characters.
- We use Stemming process to remove verbs that repeat in a sentence many times but doesn’t really add any meaning to the sentence. In Python, we are using Porter Stemmer to accomplish this.
- We then import the stop words from NLTK package and strip them off our tokenized dictionary created because words like: a, an, the, is, to, for, etc. doesn’t really add any meaningful information to our semantics.
- In the model we are constructing, we find the “term frequency” which is the number of occurrences of each word in the dataset, which is:

![Python Std Libraries](https://github.com/ssrishabh96/Machine-Learning-Email-SMS-Spam-Ham-Classifier/blob/master/images/6.png "Python Libraries")


- TF-IDF stands for Term Frequency-Inverse Document Frequency. In addition to Term Frequency we compute Inverse document frequency. If a word occurs a lot, it means that the word gives less information. In this model each word has a score, which is TF(w)*IDF(w).

![Formulae](https://github.com/ssrishabh96/Machine-Learning-Email-SMS-Spam-Ham-Classifier/blob/master/images/4.png "Formulae")

 - We combine TF-IDF with additive smoothing to account for cases when we encounter a string that we have never trained our classifier for. This technique add a number alpha to the numerator and add alpha times number of classes over which the probability is found in the denominator. The resultant equation becomes:

![Formulae](https://github.com/ssrishabh96/Machine-Learning-Email-SMS-Spam-Ham-Classifier/blob/master/images/5.png "Formulae")

 - This is done so that the least probability of any word now should be a finite number. Addition in the denominator is to make the resultant sum of all the probabilities of words in the spam emails as 1. When alpha = 1, it is called Laplace smoothing.

- For classifying a given message, first we pre-process it. For each word w in the processed messaged we find a product of P(w|spam). If w does not exist in the train dataset we take TF(w) as 0 and find P(w|spam) using above formula. We multiply this product with P(spam) The resultant product is the P(spam|message). Similarly, we find P(ham|message). Whichever probability among these two is greater, the corresponding tag (spam or ham) is assigned to the input message.

### Implementation:

- https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- https://towardsdatascience.com/introduction-to-naive-bayes-classification-4cffabb1ae54
- https://statsoft.com/textbook/naive-bayes-classifier
- https://kaggle.com/uciml/sms-spam-collection-dataset
