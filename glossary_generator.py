import preprocessing as prep
import keyword_extractor
import tfidf_evaluator


def keywords(keyword_list, file_str):
	keyword_yake = keyword_extractor.extract_keywords_yake(file_str)
	keyword_rake = keyword_extractor.extract_keywords_rake(file_str,2 ,3)
	keyword_list.extend(keyword_yake)
	keyword_list.extend(keyword_rake)
	keyword_list = list(set(keyword_list))

def remove_duplicated(list):
	list_aux = []
	[list_aux.append(item) for item in list if item not in list_aux]
	return	list_aux

def get_glossary(word_list_text, keyword_list, category):
	vocabulary = []
	vocabulary.extend(word_list_text)
	vocabulary.extend(keyword_list)
	vocabulary = remove_duplicated(vocabulary)

	glossary = tfidf_evaluator.tfidf(vocabulary, category)
	
	glossary.to_excel(f"./glossary/{category}.xlsx")  
	return glossary  

#def create_all_glossary():
#------- main ------
keyword_health = []
keyword_politics = []
keyword_sport = []

politics_file = open("./corpus/corpus_politica.txt", "w", encoding='utf-8')
health_file = open('./corpus/corpus_salud.txt', 'w', encoding='utf-8')
sport_file = open('./corpus/corpus_deporte.txt', 'w', encoding='utf-8')
prep.merge_corpus(politics_file, health_file, sport_file)
	
politics_file_str = prep.read_text_file("./corpus/corpus_politica.txt")
health_file_str = prep.read_text_file("./corpus/corpus_salud.txt")
sport_file_str = prep.read_text_file("./corpus/corpus_deporte.txt")

sport_keywords_file = open('./keywords/keywords-deporte.txt', 'w', encoding='utf-8')
politics_keywords_file = open('./keywords/keywords-politca.txt', 'w', encoding='utf-8')
health_keywords_file = open('./keywords/keywords-salud.txt', 'w', encoding='utf-8')

# get keywords
keywords(keyword_health, health_file_str)
keywords(keyword_politics, politics_file_str)
keywords(keyword_sport, sport_file_str)

health_keywords_file.write(str(keyword_health))
politics_keywords_file.write(str(keyword_politics))
sport_keywords_file.write(str(keyword_sport))

#second processing	
word_list_health = prep.process_corpus(health_file_str, True, False)
word_list_politics = prep.process_corpus(politics_file_str,True, False)
word_list_sport = prep.process_corpus(sport_file_str, True, False)

#get glossary
health_glossary = get_glossary(word_list_health, keyword_health,"salud")
politics_glossary = get_glossary(word_list_politics,keyword_politics,"politica")
sport_glossary = get_glossary(word_list_sport,keyword_sport,"deportes")

politics_file.close()
health_file.close()
sport_file.close()

health_keywords_file.close()
politics_keywords_file.close()
sport_keywords_file.close()
