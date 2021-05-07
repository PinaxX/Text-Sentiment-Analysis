from preprocess import preprocess
import pickle
from keras.models import load_model
from flask import Flask, render_template, request
import numpy as np
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 100

f = open("Extra/tokenizer.pickle", "rb")                                                            #load tokenizer
tokenizer = pickle.load(f)
f.close()
model = load_model('Extra/model.h5')                                                                #load model


app = Flask(__name__)
app.secret_key = 'some_secret'


@app.route('/', methods=['GET', 'POST'])
def index():
    sizes = [100, 100, 100, 100, 100]
    if request.method == 'GET':
        return render_template('index.html', sizes=sizes)
    else:
        comment = request.form['p1']
        comment = preprocess(comment)
        X = tokenizer.texts_to_sequences([comment])
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
        y = np.argmax(model.predict(X), axis=-1)[0]
        sizes[y] = 200
        return render_template('index.html', sizes=sizes)
    

if __name__ == '__main__':
    app.run(port = 8080)