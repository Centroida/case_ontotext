import json
import keras
from attention_with_context import AttentionWithContext
from keras import backend as K

def load_model_custom(path_weights, custom_layer_name):
    loaded_model = keras.models.load_model(path_weights, custom_objects={
                                           custom_layer_name: AttentionWithContext, "recall":recall})
    return loaded_model

def load_vocab(vocab_path='vocabulary/vocab.json'):
    '''
            Loads vocabulary for a w2v model
    '''
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
