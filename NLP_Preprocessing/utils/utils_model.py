from gensim.models.word2vec import Word2Vec, PathLineSentences
import logging
import multiprocessing
import numpy as np
import json
import gensim

def train_model(training_dir_path):
    logger = logging.getLogger("w2v_logger")
    sentences = PathLineSentences(training_dir_path)

    #logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)

    params = {"size": 100, "window": 10, "min_count": 10,
              "workers": max(1, multiprocessing.cpu_count() - 2), "sample": 1E-3}

    logger.info("training the model")

    word2vec = Word2Vec(sentences, **params)

    logger.info("saving the model")
    #Specify folder name
    word2vec.save("../models/model_n/result.model")

def train_model_on_top(model_path, more_sentences_path):
    logger = logging.getLogger("w2v_logger")
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)

    model = gensim.models.Word2Vec.load(model_path)
    logger.info("loaded the model")

    more_sentences = PathLineSentences(more_sentences_path)
    logger.info("loaded the sentences")

    #setting update to True, allows for the dictionary to accept new words
    model.build_vocab(more_sentences, update=True)

    model.train(more_sentences,
                total_examples=model.corpus_count,
                epochs=model.iter)

    model.save(model_path + "-retrained_train_test")

def test_model(path, word):
    word2vec = Word2Vec.load(path)
    print("Loaded the model")
    vec = word2vec.wv.syn0
    print(vec.shape)
    sim = word2vec.most_similar(word)
    print(sim)
    print(word2vec[word])


def update_dictionary(path):
    model = Word2Vec.load(path)
    model.build_vocab([['<unseen>'] * 10], update=True)
    model.build_vocab([['<zero>'] * 10], update=True)
    model.wv.syn0[model.wv.vocab['<zero>'].index] = np.zeros(100)
    model.wv.syn0[model.wv.vocab[
        '<unseen>'].index] = np.random.uniform(-0.25, 0.25, 100)
    model.save(path)


def extract_vocabulary(model_path, out_vocab_path):
    model = Word2Vec.load(model_path)
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    with open(out_vocab_path, 'w') as f:
        f.write(json.dumps(vocab))

update_dictionary("../models/model_1/embeddings/result.model-retrained-retrained_train_test")
#train_model_on_top("result.model-retrained", "../data/clean_data/ontotext_test.txt")
extract_vocabulary("../models/model_1/embeddings/result.model-retrained-retrained_train_test", "../models/model_1/vocabulary/voc_train_test.json")
#test_model("../models/model_1/embeddings/result.model-retrained-retrained_train_test", "<unseen>")