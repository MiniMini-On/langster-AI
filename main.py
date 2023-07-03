from tensorflow import keras
import tensorflow
import data_reader

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 50  # 예제 기본값은 50입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# 인공신경망을 제작합니다.
model = keras.Sequential([
    keras.layers.Embedding(15288, 255), #one hot vector 만들기(단어개수, 한번에 문장에 포함할 수 있는 단어 길이)
    keras.layers.GlobalAveragePooling1D(), #2차원 one hot vector를 1차원으로 압축
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

# 인공신경망을 컴파일합니다.
model.compile(optimizer='adam',
              metrics=['accuracy'],
              loss="binary_crossentropy")

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5)
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
                    validation_data=(dr.test_X, dr.test_Y),
                    callbacks=[early_stop])
score = model.evaluate(dr.test_X, dr.test_Y)

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(history)

model.save('language.h5')