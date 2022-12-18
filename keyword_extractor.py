import yake
def extract_keywords_yake(text):
    language = "es"
    max_ngram_size = 3
    deduplication_thresold = 0.5 #evitar que se repitan palabras
    deduplication_algo = 'seqm'
    windowSize = 15
    numOfKeywords = 5

    kw_extractor = yake.KeywordExtractor(lan=language, 
                                         n=max_ngram_size, 
                                         dedupLim=deduplication_thresold, 
                                         dedupFunc=deduplication_algo, 
                                         windowsSize=windowSize, 
                                         top=numOfKeywords)
    keywords = kw_extractor.extract_keywords(text)
    print(type(keywords[0]))
    return keywords
                      


from rake_nltk import Rake

def extract_keywords_rake(text):
    language = "spanish"
    min_ngram_size = 2
    max_ngram_size = 3

    rake_nltk_var = Rake( language = language
                        , min_length = min_ngram_size
                        , max_length = max_ngram_size = 3
                        )

    rake_nltk_var.extract_keywords_from_text(text)

    # keyword_extracted = rake_nltk_var.get_ranked_phrases()
    keyword_extracted = rake_nltk_var.get_ranked_phrases_with_scores()

    return keyword_extracted
