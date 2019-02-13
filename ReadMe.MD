Character Level Random Sentence Generation using LSTM Model

1. CREATE A VIRTUAL ENVIRONMENT USING CONDA:

Conda create --name ea python=2.7

2. Go to console run python main.py, it already have default filename and weight file. You can change it

Approach:
Random Sentence generation is one of the ways to see how our algorithm is learning. Natural Language Processing basic algorithms learning is based upon concept of Markov Chain, even NLP models eventually end up learning markov chain concepts but more efficiently. 

Over here we will be using a Small LSTM model which is processing 100 words at one time. It's possible to train this model in distributed environment on google cloud platform, as tf.estimator distributes this model appropriately to number of workers chosen. Similarly if the choice is to use this model as an application, its possible to do so, by putting it on ML engine on gcloud, or any other server.

For embeddings of words: currently we are just using one hot encoding vector, its completely possible to use gensim/word2vec model, that will improve the mode as the context learnt will be much better.

I didn't get chance to train this model for more than 5 epochs, I am hoping that it will improve once it has been trained for more than 100 epochs.

Steps in approach:
---------------------
1. Read the text file.
2. Create a dictionary to convert word in embeddings format(for now one hot encoding), its possible to either use NY times trained word2vec embeddings/gensim embeddings, or considering Gutenberg have big dataset, we can also fine-tune word2vec of our own, those embeddings will be able to capture the context at best.
3. Create a small LSTM Model, we can also try to deepen it considering what level of accuracy we want.
4. Loss function is categorical cross entropy, with Adam optimizer.
5. Test the model by seeding a random word, and it will try to complete it.



