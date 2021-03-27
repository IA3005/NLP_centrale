# NLP_centrale

This exercise was done by : 

- Azza BEN FARHAT (azza.ben-farhat@student.ecp.fr)
- Imen AYADI (imen.ayadi@student.ecp.fr)
- Zaineb Letaief (zaineb.letaief@student.ecp.fr)


To solve this aspect-based polarities of opinions classification exercise, 
we implemented a classifier based on the Bert Base Cased model and used this tutorial: 
https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/.



We used the opinion sentences and the target terms as inputs to the model. 
In fact, when we tokenize these two features, we put a separator token "SEP" between both and 
used two different types of token_type_ids to make sure the model differentiates both features. 
Then, we encode them using a padding to max_length, which ads 0s to the tokens to reach the 
max_length for padding, as the bert model needs tokens of same length (in this case, the length is max_length).

The model is the pretrained Bert Base Cased model to which we add a dropout layer with p=0.3, and a linear layer which represents the output layer.
It uses the as inputs: the inputs ids, the attention mask and the token type ids.



The max_length parameter used is 100, with a batch size of 9 and 10 epochs.
The optimizer used is AdamW with a learning rate equal to 2e-5.

The accuracy that we got on the dev dataset is 82.71%. 
Although the training dataset is very imbalanced (positive: 70.19%, neutral: 3.85% and negative: 25.94%), the model reaches a high accuracy.

Before the BERT model, we have implemented classifiers based on non-deep networks. You can find the code in "classifier_1.py" and "classifier_2.py". 
To run them, you need only to replace "from classifier import Classifier" by "from classifier_1 import Classifier" or "from classifier_2 import Classifier" in "tester.py"
For both classifiers 1 and 2, we started by preprocessing the data (tokenize, downcase, remove stopwords except "no", "nor", "nor" to keep the meaning, stemming). 
Then, we created the BOW vector and transformed it with the tfidf scores. 
* For "classifier_1", we used a linear Support Vector Classifier and we get a test accuracy of 79.26% on the dev dataset.
* For "classifier_2", we used the multinomial Naive Bayes classifier and we get a test accuracy of 77.93% on the dev dataset.

