﻿# ShayaSimpleSentimentTask

I'm going to write this document in the order that the code should run in:  
  
**1- Preprocess:**  
File: preprocess.py  
Input: Dataset/DeepSentiPers-original.csv  
Output: Dataset/Preprocessed_Data  
Description: Keeps only some punctuation and Farsi characters. Does text normalization (and stemming) usig Hazm. Adds space before punctuation.  
  
**2- Train test split:**  
File: train_test_split.py  
Input: Dataset/Preprocessed_Data  
Output: Dataset/Preprocessed_Train_Data.csv, Dataset/Preprocessed_Test_Data.csv  
Description: Randomly selects 80% of data for training and leaves the rest for test.  
  
**3- Persian dictionary:**  
File: process_persian_dict.py  
Input: Extra/فایل-فرهنگ-فارسی.xlsx (Downloaded from [This link](https://bigdata-ir.com/%d9%81%d8%a7%db%8c%d9%84-%d9%81%d8%b1%d9%87%d9%86%da%af-%d9%81%d8%a7%d8%b1%d8%b3%db%8c-%d8%a8%d8%a7-%d9%81%d8%b1%d9%85%d8%aa-csv-%d9%82%d8%a7%d8%a8%d9%84-%d8%a8%d8%a7%d8%b1%da%af%d8%b0%d8%a7%d8%b1/))   
Output: Extra/Persian_Dictionary.pickle  
Description: Creates a dictionary which later is used to generate augmented training data. Gets raw dictionary file as input, preprocesses the text, removes antonyms and saves the result as a dictionary (with words as keys and synonyms as values).  
  
**4- Data augmentation:**  
File: data_augmentation.py  
Input: Dataset/Preprocessed_Train_Data.csv, Extra/Persian_Dictionary.pickle  
Output: Dataset/Augmented_Train_Data.csv  
Description: Uses the Persian dictionary to generate new data in minority classes (-2, -1, 2), by replacing words from existing data with their synonyms.  
