import seaborn as sns
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, CuDNNLSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import h5py

def load_model_h5():
	vocabulary_size = 977094
	max_words = 25
	#previous batch size was 128
	batch_size = 508
	lstm_neurons = 128
	dense_neurons = 0
	embedding_size = 45
	dropout_rate = 0.3
	model = Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
	model.add(Dropout(dropout_rate))
	model.add(CuDNNLSTM(lstm_neurons, input_shape=(X_train.shape[1:])))
	model.add(BatchNormalization())
	model.add(Dropout(dropout_rate))
	model.add(Dense(2, activation='sigmoid'))
	return model

pickle_in = open("D:/Python/Data/SENTIMENT140_set_keras.pickle", "rb")
X_train, X_test, y_train, y_test = pickle.load(pickle_in)
print('Data loaded')

model = load_model_h5()
model.load_weights("C:/Felix Ordner/Python/NNN/SENTIMENT140-2-Dropout-0.3-DropRate-1-LSTM-128-neurons0-Dense-0-dense_neurons-45-embed_size-25-max_words-1559031797.7983096-time.h5")
print('Model loaded')

yhat_class = model.predict_classes(X_test, verbose=0)


sns.heatmap(confusion_matrix(y_test,yhat_class),annot=True,fmt='d', yticklabels=True, cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()