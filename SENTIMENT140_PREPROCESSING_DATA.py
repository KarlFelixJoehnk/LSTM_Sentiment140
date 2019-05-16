import pandas as pd
import random
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

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

def create_train_test_set(fs):
	X = []
	y = []

	for feat in fs:
		#filter out neutral sentiment
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

	tk = Tokenizer(lower = True)
	tk.fit_on_texts(X)
	X_seq = tk.texts_to_sequences(X)
	for feature in X_seq[:3]:
		print(feature)

	X_pad = pad_sequences(X_seq, maxlen=25, padding='post')
	for feature in X_pad[:3]:
		print(feature)
	vocabulary_size = len(tk.word_counts.keys())+1
	print(vocabulary_size)

	X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.1, random_state = 1)

	return X_train, X_test, y_train, y_test

#create a dataframe
df = pd.read_csv("C:/Felix Ordner/Python/NNN/Data/SENTIMENT140.csv", encoding='latin1', names = ["Sentiment", "ID", "Date", "Flag", "User", "Text"])
print(df.Text.head(5))

#create a lemmatized column within the dataframe
df['text_lemmatized'] = df.Text.apply(lemmatize_text)
print(df.text_lemmatized.head(5))

#create a list of lists featureset
my_featureset = create_featuresets(df)
#print(my_featureset[:5])
my_featureset = filer_unnec_words(my_featureset)

random.seed(123)
random.shuffle(my_featureset)

X_train, X_test, y_train, y_test = create_train_test_set(my_featureset)
print(len(X_train), len(X_test))

with open('D:/Python/Data/SENTIMENT140_set_keras.pickle', 'wb') as f:
	pickle.dump([X_train, X_test, y_train, y_test], f)

