import os
import preprocessing

import pandas as pd

def code_to_class(code):
	if code == 0:
		return 'deportes'
	if code == 1:
		return 'salud'
	return 'politica'

results_df = pd.read_excel("./corpus/test_results.xlsx")
clasification_path = './clasification/'
results_df.sort_values("predicted_class")

sport_df = results_df.loc[results_df['predicted_class'] == 0].sort_values("probability", ascending=False,ignore_index=True)
health_df = results_df.loc[results_df['predicted_class'] == 1].sort_values("probability",ascending=False,ignore_index=True)
politics_df = results_df.loc[results_df['predicted_class'] == 2].sort_values("probability",ascending=False,ignore_index=True)

def get_file_name(predicted_class, name):
	if predicted_class == 0:
		row = sport_df.loc[sport_df['name'] == name]
	elif predicted_class == 1:
		row = health_df.loc[health_df['name'] == name]
	else:
		row = politics_df.loc[politics_df['name'] == name]
	index = row.index._data[0]
	new_name = f"{index}-{name}"
	return new_name

for root, dirs, files in os.walk("./corpus/test", topdown=False):
	for name in files:
		predicted_class = results_df.loc[results_df['name'] == name]['predicted_class']
		predicted_class = predicted_class.item()
		file_str = preprocessing.read_text_file(os.path.join(root, name))
		new_name = get_file_name(predicted_class, name)
		new_file = open(f"{clasification_path}{code_to_class(predicted_class)}/{new_name}", "w", encoding='utf-8')
		new_file.write(file_str)
		new_file.close()

