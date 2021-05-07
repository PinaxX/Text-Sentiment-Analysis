import pandas as pd
import pickle

data = pd.read_csv('Dataset/Preprocessed_Train_Data.csv')

with open('Extra/Persian_Dictionary.pickle', 'rb') as handle:
    persian_dict = pickle.load(handle)                                                      #loading persian dict

augmentation_count_dict = {-2: 3, -1: 2, 2: 1}                                              #number of synonyms (based on class distribution)
augmented_data = []

for label in [-2, -1, 2]:                                                                   #minority classes
    for text in data.loc[data['Label'] == label][:len(data.loc[data['Label'] == -2])]['Text'].tolist():
        try:
            tokens = text.split()
            for i, word in enumerate(tokens):
                if word not in ['?', '.', '!', ',', '؟', '،'] and word in persian_dict:
                    similar_words = persian_dict[word][:min(len(persian_dict[word]), augmentation_count_dict[label])]
                    for similar_word in similar_words:
                        tmp = tokens
                        tmp[i] = similar_word
                        augmented_data.append([" ".join(tmp), label])
        except:
            pass

aug_df = pd.DataFrame(augmented_data, columns=['Text', 'Label'])
df = pd.concat([data, aug_df], ignore_index=True)
df.to_csv('Dataset/Augmented_Train_Data.csv', index=False)