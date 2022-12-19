import yake
def extract_keywords_yake(text):
    language = "es"
    max_ngram_size = 3
    deduplication_thresold = 0.5 #evitar que se repitan palabras
    deduplication_algo = 'seqm'
    windowSize = 15
    numOfKeywords = 10

    kw_extractor = yake.KeywordExtractor(lan=language, 
                                         n=max_ngram_size, 
                                         dedupLim=deduplication_thresold, 
                                         dedupFunc=deduplication_algo, 
                                         windowsSize=windowSize, 
                                         top=numOfKeywords)
    keywords = kw_extractor.extract_keywords(text)
    keywords_list = []
    for item in keywords:
        keywords_list.append(item[0])
    return keywords_list
                      


from rake_nltk import Rake

def extract_keywords_rake(text, min_size, max_size):
    language = "spanish"
    min_ngram_size = min_size
    max_ngram_size = max_size

    rake_nltk_var = Rake( language = language
                        , min_length = min_ngram_size
                        , max_length = max_ngram_size
                        )

    rake_nltk_var.extract_keywords_from_text(text)

    keyword_extracted = rake_nltk_var.get_ranked_phrases()
    #keyword_extracted = rake_nltk_var.get_ranked_phrases_with_scores()
    return keyword_extracted
