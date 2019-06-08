import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, CuDNNLSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import time
import h5py

pickle_in = open("D:/Python/Data/SENTIMENT140_set_keras.pickle", "rb")
X_train, X_test, y_train, y_test = pickle.load(pickle_in)

#pad length must be the same as input_length!!!!
vocabulary_size = 977094
max_words = 25
#previous batch size was 128
batch_size = 508
lstm_neurons = 128
dense_neurons = 0
#X_train1 = X_train[batch_size:]
#y_train1 = y_train[batch_size:]
#X_valid = X_train[:batch_size]
#y_valid = y_train[:batch_size]
embedding_size = 45
dropout_rate = 0.3

#before 128
NAME = "SENTIMENT140-{}-Dropout-{}-DropRate-{}-LSTM-{}-neurons{}-Dense-{}-dense_neurons-{}-embed_size-{}-max_words-{}-time".format(2,dropout_rate,1,lstm_neurons,0,dense_neurons,embedding_size, max_words, time.time())

print(NAME)
#see while training cmd--> tensorboard --logdir=logs/
tensorboard = TensorBoard(log_dir='newlogs/{}'.format(NAME))

model = Sequential()

model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(Dropout(dropout_rate))

model.add(CuDNNLSTM(lstm_neurons, input_shape=(X_train.shape[1:])))
model.add(BatchNormalization())
#previous Dropout was 0.2 
model.add(Dropout(dropout_rate))

model.add(Dense(2, activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size,epochs=4, callbacks=[tensorboard])

#FOR creating a confusion matrix 
#https://www.youtube.com/watch?v=km7pxKy4UHU

# Save the weights
model.save_weights('{}.h5'.format(NAME))

# Save the model architecture
with open('{}_architecture.json'.format(NAME), 'w') as f:
    f.write(model.to_json())