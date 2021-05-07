import pandas as pd

data = pd.read_csv('Dataset/Preprocessed_Data.csv')

train_data = data.sample(frac=0.8, random_state=42)
test_data = data.loc[~data.index.isin(train_data.index)]

train_data.to_csv('Dataset/Preprocessed_Train_Data.csv', index=False)
test_data.to_csv('Dataset/Preprocessed_Test_Data.csv', index=False)