import tensorflow as tf
from keras import backend as K
import utils.utils_iterator as utils_iterator
import utils.utils_wv as utils_wv
import utils.utils_model as utils_model
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Input, LSTM, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import LearningRateScheduler

batchsize = 512
seed = 7
np.random.seed(seed)

model = utils_model.model_stacked_blstsms_with_pos_attn()

#model.summary()

model.compile(loss='binary_crossentropy',
		optimizer=Adam(),
		metrics=['accuracy'])

model.fit_generator(generator=utils_iterator.generate_data('D:\\git\\ontotext\\ontotext\\NLP_Preprocessing\\data\\clean_data\\ontotext_train_labeled.txt', 
                        batchsize, 
                        input_size=75, 
                        with_positions=True),  
                    validation_data=utils_iterator.generate_data(
                        'D:\\git\\ontotext\\ontotext\\NLP_Preprocessing\\data\\clean_data\\ontotext_test_labeled.txt', 
                        batchsize, 
                        input_size=75, 
                        with_positions=True), 
                    steps_per_epoch= 89452//(batchsize),
                    validation_steps= 20591//(batchsize), 
                    workers=4, 
                    epochs=70,
                    verbose=2)
