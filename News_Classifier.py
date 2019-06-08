import pandas as pd
import random
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, CuDNNLSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import pickle

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

def filer_unnec_words(fs):
	#filter out @words
	for features in fs:
		#string out of range
		features[1] = [x for x in features[1] if x[0]!='@']
	#for features in fs[:5]:
	#	print(features)
	return fs

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

def create_featuresets(df):
	featureset = []
	for row in df.values:
		featureset.append([row[0],row[6]])

	return featureset

def create_news_featureset_X(df):
	X = []
	for row in df.values:
		X.append(row[0])

	return X	

def create_feature_lists(fs):
	X = []
	y = []

	for feat in fs:
		#filter out neutral sentiment
		#actually there is none in this dataset
		if feat[0] != 2:
			if feat[0] == 4:
				X.append(feat[1])
				y.append(1)
			elif feat[0] == 0:
				X.append(feat[1])
				y.append(feat[0])

	print("negatives: {}".format(y.count(0)))
	print("positives: {}".format(y.count(1)))
	print("neutral: {}".format(y.count(2)))
	
	for feature in X[:3]:
		print(feature) 
	return X, y

def nlp_processing(X, X_news):
	tk = Tokenizer(lower = True)
	tk.fit_on_texts(X)
	for feature in X_news[:3]:
		print(feature)
	X_seq = tk.texts_to_sequences(X_news)
	for feature in X_seq[:3]:
		print(feature)
	X_pad = pad_sequences(X_seq, maxlen=25, padding='post')
	for feature in X_pad[:3]:
		print(feature)
	vocabulary_size = len(tk.word_counts.keys())+1
	print(vocabulary_size)
	
	return X_pad

#Where the final df will be saved to
export_data = "D:/Python/Data/US_Financial_News/Apple_News_with_pred_Sentiment.csv"

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

#create a dataframe from training data for dictionary creation later on
df = pd.read_csv("C:/Felix Ordner/Python/NNN/Data/SENTIMENT140.csv", encoding='latin1', names = ["Sentiment", "ID", "Date", "Flag", "User", "Text"])
print(df.Text.head(5))
#load news data 
df_news = pd.read_csv("D:/Python/Data/US_Financial_News/Filtered_News_2018_01_05.csv", encoding='utf-8')
print(df_news.Title.head(5))

#create a lemmatized column within the dataframe train
df['text_lemmatized'] = df.Text.apply(lemmatize_text)
print(df.text_lemmatized.head(5))
#create a lemmatized column within the dataframe news
df_news['text_lemmatized'] = df_news.Title.apply(lemmatize_text)
print(df_news.text_lemmatized.head(5))

############PREPROCESS TRAINING DATA FOR DICTIONARY
#create a list of lists featureset
my_featureset = create_featuresets(df)
#print(my_featureset[:5])
my_featureset = filer_unnec_words(my_featureset)
random.seed(123)
random.shuffle(my_featureset)
X, y = create_feature_lists(my_featureset)
#Make some space in RAM
my_featureset = None
############PREPROCESS NEWS DATA
X_news = create_news_featureset_X(df_news)

text_pad = nlp_processing(X, X_news)
df = None
y = None

#load the training data for setting up the model input data shape
pickle_in = open("D:/Python/Data/SENTIMENT140_set_keras.pickle", "rb")
X_train, X_test, y_train, y_test = pickle.load(pickle_in)
#Load model for prediction
model = load_model_h5()
model.load_weights("C:/Felix Ordner/Python/NNN/SENTIMENT140-2-Dropout-0.3-DropRate-1-LSTM-128-neurons0-Dense-0-dense_neurons-45-embed_size-25-max_words-1559031797.7983096-time.h5")
print('Model loaded')


yhat_class = model.predict_classes(text_pad, verbose=0)
yhat = model.predict(text_pad, verbose=0)
print("Prediction for {} is {} with a probability of {}".format(X_news[1],yhat_class[1], yhat[1]))

df_news['pred_Sentiment'] = yhat_class

dummy_df = pd.DataFrame(yhat, columns =['deg_neg', 'deg_pos'])

df_news = pd.concat([df_news,dummy_df], axis=1)

df_news.to_csv(export_data, encoding='utf-8', index=None)

