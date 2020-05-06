from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential, layers
from keras.datasets import imdb

import matplotlib.pyplot as plt

top_words = 5000
max_words = 500
embedding_dim = 64

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

print("Training Dataset" + str(len(X_train)))
print("Test Dataset" + str(len(X_test)))


## Model
model = Sequential()
model.add(layers.Embedding(top_words, embedding_dim, input_length=max_words))
model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(250, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 5

history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=512,
                    validation_data=(X_test, y_test))

_, acc = model.evaluate(X_train, y_train)
_, acc_test = model.evaluate(X_test, y_test)

print(acc)
print(acc_test)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
