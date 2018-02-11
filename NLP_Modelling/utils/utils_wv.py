from gensim.models.word2vec import Word2Vec, PathLineSentences
import logging 
import multiprocessing
import numpy as np
from keras.layers import Embedding

def update_model():
    model = Word2Vec.load('../models/namehere.model')
    model.build_vocab([['<unseen>']*10], update=True)
    model.build_vocab([['<zero>']*10], update=True)
    model.wv.syn0[model.wv.vocab['<zero>'].index] = np.zeros(100)
    model.wv.syn0[model.wv.vocab['<unseen>'].index] = np.random.uniform(-0.25, 0.25, 100)
    model.save('../models/namehere.model')
    print(model['<zero>'])
    print(model['<unseen>'])

def word2vec_embedding_layer(embeddings_path='../NLP_Preprocessing/models/model_1/embeddings/result.model-retrained-retrained_train_test.wv.vectors.npy'):
    weights = np.load(open(embeddings_path, 'rb'))
    print(weights.shape)
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights], trainable=False)
    return layer
