import pandas as pd
import re
from hazm import *
import pickle

def normalize(text):
    text = text.strip()
    text = normalizer.normalize(text)
    text = text.replace(' ', '')
    text = text.replace('\u200c', ' ')                                                              #half space to space
    text = re.sub(r"[^آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی؟،?.! ,]+", "", text)
    return text

df = pd.read_excel('Extra/فایل-فرهنگ-فارسی.xlsx', header=None)                                           #loading original dictionary
words = df[0].tolist()
synonyms = df[1].tolist()

normalizer = Normalizer()
persian_dict = {}

for i, word in enumerate(words):
    try:
        normalized_word = normalize(word)
        if normalized_word not in persian_dict:
            syns = []
            for syn in synonyms[i].split('&')[0].split('،'):                                        #seperating antonyms and synonyms
                syns.append(normalize(syn))
            persian_dict[normalized_word] = syns
    except:
      pass

with open('Extra/Persian_Dictionary.pickle', 'wb') as handle:
    pickle.dump(persian_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)                             #saving dictionary