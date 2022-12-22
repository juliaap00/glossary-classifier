import os
import preprocessing

import pandas as pd

DEPORTE = 0
SALUD = 1
POLITICA = 2

df_train = pd.DataFrame()
df_train["document"] = []
df_train["class"] = []
df_train["name"] = []

df_test = pd.DataFrame()
df_test["document"] = []
df_test["class"] = []
df_test["name"] = []


for root, dirs, files in os.walk("./corpus/", topdown=False):
	for name in files:
		if ('train\\' in root or 'test\\' in root):
			file_str = preprocessing.read_text_file(os.path.join(root, name))
			file_str = preprocessing.preprocess_corpus(file_str)
			file_str = preprocessing.process_corpus(file_str, False, True)
			
			if 'deportes' in root:
				category = DEPORTE
			elif 'salud' in root:
				category = SALUD
			elif 'politica' in root:
				category = POLITICA

			if 'train' in root:
				df_train.loc[len(df_train)] = [file_str, category, name]
			elif 'test' in root:
				df_test.loc[len(df_test)] = [file_str, category, name]


df_train_shuffled = df_train.sample(frac=1).reset_index(drop=True)
df_test_shuffled = df_test.sample(frac=1).reset_index(drop=True)

df_train_shuffled.to_excel(f"./corpus/train.xlsx")
df_test_shuffled.to_excel(f"./corpus/test.xlsx")



