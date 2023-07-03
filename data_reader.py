"""
Author : Byunghyun Ban
Date : 2020.07.17.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import random
import csv
import time
import konlpy
from konlpy.tag import Okt
import pickle
import re

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'matplotlib'])
    try:
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        time.sleep(2)
        from matplotlib import pyplot as plt

try:
    import numpy as np
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'numpy'])
    try:
        import numpy as np
    except ModuleNotFoundError:
        time.sleep(2)
        import numpy as np


# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self):
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_data()

        # 데이터 읽기가 완료되었습니다.
        # 읽어온 데이터의 정보를 출력합니다.
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')

    def read_data(self):
        okt = Okt()
        hangle = re.compile('[^ ㄱ-ㅣ가-힣]+')
        data = []
        
        # filename = "data/" + os.listdir("data")[1]
        # print(filename)
        # file = open(filename, 'rt', encoding='UTF8')

        # for line in file:
        #     splt = line.split("|")
        #     if "0" in splt[1]:
        #         y = 0.0
        #     elif "1" in splt[1]:
        #         y = 1.0
        #     # x = splt[0].strip()
        #     x = splt[0]
        #     if (x, y) not in data:
        #         data.append((x, y))

        filename = "data/" + os.listdir("data")[0]
        print(filename)
        file = open(filename, 'r', encoding='UTF8')
        rdr = list(csv.reader(file))
        for line in rdr[1:]:
            # print(line)
            line = line[0].split('\t')
            if "0" in line[1]:
                y = 1.0
            elif "1" in line[1]:
                y = 0.0
            # x = splt[0].strip()
            x = line[0]
            if (x, y) not in data:
                data.append((x, y))
        file.close()
        random.shuffle(data)

        X = []
        Y = []

        for el in data:
            X.append(el[0])
            Y.append(el[1])

        Y = np.asarray(Y)

        X = [hangle.sub('', word) for word in X]
        X = [okt.morphs(word, stem = True, norm = True) for word in X]
        
        tokenizer = keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(X)
        X = tokenizer.texts_to_sequences(X)
        X = keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=255)

        train_X = X[:int(0.8*len(X))]
        train_Y = Y[:int(0.8*len(Y))]
        test_X = X[int(0.8*len(X)):]
        test_y = Y[int(0.8*len(Y)):]
        print(test_X)
        print(test_y)
        # print(tokenizer.word_index)
        with open('data_dict.pkl', 'wb') as f:
            pickle.dump(tokenizer.word_index, f)
        return train_X, train_Y, test_X, test_y


def draw_graph(history):
    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Loss History")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS Function")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("train_history.png")

    train_history = history.history["acc"]
    validation_history = history.history["val_acc"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Accuracy History")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("accuracy_history.png")
