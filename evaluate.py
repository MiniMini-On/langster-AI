from tensorflow import keras
import tensorflow as tf
import konlpy
from konlpy.tag import Okt
import pickle
import re
import data_reader


dr = data_reader.DataReader()

model = tf.keras.models.load_model('language.h5')
score = model.evaluate(dr.test_X, dr.test_Y)
print('Test loss:', score[0])
print('Test accuracy:', score[1])