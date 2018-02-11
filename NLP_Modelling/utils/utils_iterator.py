import threading
from gensim.models.word2vec import Word2Vec
import json
import random
import numpy as np
import sklearn.preprocessing

class threadsafe_iter(object):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
        word2idx = data
        idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word

def standartize_word_indeces(mean_length, array, index_to_use):
    array = np.asarray(array)

    zero_word = np.array([index_to_use])

    if array.shape[0] > mean_length:
        array = array[:mean_length]
    elif array.shape[0] < mean_length:
        while array.shape[0] < mean_length:
            array = np.concatenate([array, zero_word])      
    
    return array


def get_word_index(word, word2idx, default_index):
    if word not in word2idx:
        return default_index

    idx = word2idx[word]
    return idx


@threadsafe_generator
def generate_data(dir_path, batch_size=1, input_size=50, onehot=False, with_positions=False):
    word2idx, idx2word = load_vocab('../NLP_Preprocessing/models/model_1/vocabulary/voc_train_test.json')

    unseen_index = word2idx['<unseen>']
    zero_index = word2idx['<zero>']

    first_org_index = word2idx['<firstorganization>']
    second_org_index = word2idx['<secondorganization>']

    print(first_org_index)
    label_binarizer = sklearn.preprocessing.LabelBinarizer()

    # Variable to keep track on iterated tweets, so we do not include them in batch
    iterated_items = 0
    with open(dir_path) as f:
        lines = f.readlines()

        random.shuffle(lines)

        print(len(lines))
        while True:

            batch_list = []
            batch_labels = []
            batch_positions = []
            for b in range(batch_size):
                if iterated_items == len(lines):
                    iterated_items = 0
                    random.shuffle(lines)

                tweet = lines[iterated_items]

                iterated_items += 1

                tokenized = tweet.split(' ')
                label = tokenized.pop()

                #get_word_embedding_validedding(word, word2idx, subs_vector, model)
                sample = [get_word_index(word, word2idx, unseen_index) for word in tokenized]

                sample = standartize_word_indeces(input_size, sample, zero_index)

                if onehot:
                    first = np.zeros(input_size, dtype="int8")
                    second = np.zeros(input_size, dtype="int8")
                    np.put(first, np.argwhere(sample==first_org_index), 1)
                    np.put(second, np.argwhere(sample==second_org_index), 1)
                    batch_positions.append(np.asarray([first, second]))
                else:
                    positions = np.zeros(input_size, dtype="int8")
                    np.put(positions, np.argwhere(sample==first_org_index), 1)
                    np.put(positions, np.argwhere(sample==second_org_index), 2)
                    batch_positions.append(positions)

                batch_list.append(sample)
                batch_labels.append(int(label))

            X = np.asarray(batch_list)
            Y = np.asarray(batch_labels)
            P = np.asarray(batch_positions)
            if with_positions:
                yield [X, P], Y
            else:
                yield X, Y
