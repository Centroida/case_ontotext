import tensorflow as tf
from keras import backend as K
from keras import layers
import utils.utils_iterator as utils_iterator
import utils.utils_wv as utils_wv
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import utils.attention_with_context as a

def model_stacked_lstsms():
    model_input = Input(shape=(75,))
    embeddings = utils_wv.word2vec_embedding_layer()(model_input)

    #z = Dropout(0.5)(embeddings)
    z = LSTM(64, return_sequences=True)(embeddings)
    z = LSTM(64)(z)
    z = Dense(20, activation="relu")(z)
    z = Dropout(0.5)(z)
    z = Dense(1, activation="sigmoid")(z)
    
    return Model(model_input, z, name='stacked_lstsms')

def model_stacked_lstsms_with_pos():
    sentence = Input(shape=(75,))
    position = Input(shape=(75,))

    embeddings = utils_wv.word2vec_embedding_layer()(sentence)

    pos_embeddings = layers.Embedding(1,3, input_length=75, trainable=True)(position)
    x = layers.concatenate([embeddings, pos_embeddings])
    
    z = LSTM(64, return_sequences=True)(x)
    z = LSTM(64)(z)
    z = Dense(20, activation="relu")(z)
    z = layers.Dropout(0.5)(z)
    z = Dense(1, activation="sigmoid")(z)
    return Model([sentence, position], z, name='stacked_lstm_pos')

def model_stacked_blstsms_with_pos_attn():
    sentence = Input(shape=(75,))
    position = Input(shape=(75,))

    embeddings = utils_wv.word2vec_embedding_layer()(sentence)

    pos_embeddings = layers.Embedding(1,3, input_length=75, trainable=True)(position)
    x = layers.concatenate([embeddings, pos_embeddings])
    z = layers.Dropout(0.4)(x)
    z = Bidirectional(LSTM(256, return_sequences=True))(z)
    z = Bidirectional(LSTM(256, return_sequences=True))(z)
    z = a.AttentionWithContext()(z)
    z = Dense(10, activation="relu")(z)
    z = layers.Dropout(0.5)(z)
    z = Dense(1, activation="sigmoid")(z)

    return Model([sentence, position], z, name='stacked_lstm_pos')