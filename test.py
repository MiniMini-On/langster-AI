from tensorflow import keras
import tensorflow as tf
import konlpy
from konlpy.tag import Okt
import pickle
import re

with open('data_dict.pkl', 'rb') as f:
    mydict = pickle.load(f)
okt = Okt()
hangle = re.compile('[^ ㄱ-ㅣ가-힣]+')
new_model = tf.keras.models.load_model('language.h5')
X = ['너 진짜 또라이 아니냐', '오늘은 라1면을 먹지 않고 집밖에 나가서 놀았다.' ]
X = [hangle.sub('', word) for word in X]
X = [okt.morphs(word, stem = True, norm = True) for word in X]
tokenizer = keras.preprocessing.text.Tokenizer(filters='')
tokenizer.word_index = mydict
# tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=255)

print(X)
res = new_model.predict(X)
print(res)
# print(tokenizer.word_index)

    
