import pandas as pd
import re
from hazm import *

def preprocess(text):
    text = text.strip()
    text = normalizer.normalize(text)                                                               #normalizing
    text = re.sub(r"([?.!,؟،])", r" \1 ", text)                                                     #adding space before punctuation
    text = re.sub(r"[^آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی؟،?.!,]+", " ", text)                     #keeping only Farsi characters
    text = text.strip()
    # tmp = word_tokenize(text)                                                                     #stemming
    # stemmed = []
    # for word in tmp:
    #   stemmed.append(stemmer.stem(word))
    # text = " ".join(stemmed)
    return text

normalizer = Normalizer()
# stemmer = Stemmer()

raw_data = pd.read_csv('Dataset/DeepSentiPers-original.csv', header=None)                           #loading raw data
preprocessed_data = []

for text, label in zip(raw_data[0].tolist(), raw_data[1].tolist()):
    preprocessed_data.append([preprocess(text), label])

df = pd.DataFrame(preprocessed_data, columns=['Text', 'Label'])
df.to_csv('Dataset/Preprocessed_Data.csv', index=False)                                             #saving preprocessed data