# glossary-classifier
Implements a glossary classifier that process spanish news of three topics: Health, Sports and Politics, generates a glossary and organices news into one of these catoegries taking in account the similarity of their contents within the glossary.
## Structure

#### Preprocessing.py: 
contains functions to preprocess text and read files 

#### Process_corpus.py: 
script that process each category corpus and mergen them into two separate excels files contained on /corpus, being test.xlsx and train.xlsx
#### Keyword_extractor.py: 
contains functions calling YAKE and RAKE for keyword extraction
#### Tfidf_evaluator.py: 
contains function that calls to Sklesanr tf-idf
#### Glossary_generator.py: 
script that generates a list of keywords and glossary for each of the categories
#### Naive_Bayes.ipynb: 
Naive Bayes model Jupyter Notebook
#### SVM.ipynb: 
SVM model Jupyter Notebook
#### File_organizer.py: 
Script that organices files into folders taking in account their predicted class by the SVM model

-	### /classification	  
contains the result of calling File_organizer.py and contains the original documents organiced by similarity on their respectives predicted folders
- ### /corpus
contains the original and processed documents and a excel file with the results of the SVM model, test_results.xlsx
- ### /glossary
contains the tree original glossary generated, one for each category, and their processed versions
- ### /keywords
contains three text files with the keyword lists of each category

## Requeriments
pip install yake

pip install rake-nltk

pip install pandas

pip install sklearn

pip install scikit-learn

pip install openpyxl
