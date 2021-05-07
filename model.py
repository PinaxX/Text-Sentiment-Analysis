import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, GlobalMaxPool1D, LSTM, Bidirectional
from keras.metrics import categorical_accuracy

WORD_EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 100
batch_size = 64
epochs = 10

train_data = pd.read_csv('Dataset/Augmented_Train_Data.csv')
train_data = train_data[pd.notnull(train_data['Text'])]
test_data = pd.read_csv('Dataset/Preprocessed_Test_Data.csv')
test_data = test_data[pd.notnull(test_data['Text'])]

token_corpus = [[word for word in sentence.split()] for sentence in list(train_data.Text)]
w2v_model = Word2Vec(token_corpus, min_count=2, window=5, vector_size=WORD_EMBEDDING_DIM, sg=1)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(train_data.Text))

X_train = tokenizer.texts_to_sequences(train_data.Text)
X_test = tokenizer.texts_to_sequences(test_data.Text)
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')

y_train = to_categorical(np.array(list(train_data.Label)) + 2)
y_test = np.array(list(test_data.Label)) + 2

vocab_size = len(tokenizer.word_index)

embedding_matrix = np.zeros((vocab_size + 1, WORD_EMBEDDING_DIM))
for w, i in tokenizer.word_index.items():
    if i < vocab_size:
        try:
            vector = w2v_model.wv.get_vector(w)
            embedding_matrix[i] = vector
        except:
            pass
    else:
        break

lstm_model = Sequential()
lstm_model.add(Embedding(vocab_size + 1, WORD_EMBEDDING_DIM, weights=[embedding_matrix],
                input_length=MAX_SEQUENCE_LENGTH, trainable=False))
lstm_model.add(Bidirectional(LSTM(300, return_sequences=True)))
lstm_model.add(GlobalMaxPool1D())
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(300, activation="relu"))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(5, activation='softmax'))

lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
hist = lstm_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1)

y_pred = np.argmax(lstm_model.predict(X_test), axis=-1)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}\n'.format(f1_score(y_test, y_pred, average='weighted')))

f1 = f1_score(y_test, y_pred, average=None)
cm = confusion_matrix(y_test, y_pred)
for i in range(5):
    print('Class ' + str(i) + ' ---> F1: ' + '{:0.2f}'.format(f1[i]) + '\tConfusion row:\t' + '\t'.join(map(str, cm[i])))