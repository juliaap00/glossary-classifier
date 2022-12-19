import pandas as pd
import glob
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import minmax_scale

def tfidf(myvocabulary, category):

	directory_path = f"corpus/train/{category}/"

	corpus = glob.glob(f"{directory_path}*.txt")
	corpus_titles = [Path(text).stem for text in corpus]
	print()

	#myvocabulary = ['de','en','el','la','que','un','con','los','del','para','se','por','una','su','las','no','al','es','lo','más','pero','ha','año','puntos','dos','final','muy','españa','como','le','fue','este','equipo','sus','equipos','ya','juegos','partido','mejor','cuando','tras','si','temporada','sin','tiene','arco','porque','desde','ante','mundo']

	vectorizer = TfidfVectorizer(input='filename', vocabulary = myvocabulary, ngram_range=(1,3))
	vect_matrix = vectorizer.fit_transform(corpus)
	print(vect_matrix)
	words = vectorizer.get_feature_names_out()
	docs = vect_matrix.toarray()
	words_tfidf = []

	for i in range (0, len(docs[0])):
		word_val = 0
		for doc in docs:
			word_val += doc[i]
		words_tfidf.append(word_val)

	tfidf_df = pd.DataFrame(docs, index=corpus_titles, columns=words)

	tfidf_df.loc['00 Doc Freq'] = minmax_scale((tfidf_df).sum(), feature_range=(0,1))

	tfidf_df = tfidf_df.sort_index()
	tfidf_df = tfidf_df.sort_values(by = '00 Doc Freq', axis = 1, ascending = False)
	tfidf_df = tfidf_df.transpose()

	#print(tfidf_df[:50])
	return tfidf_df[:-1]
