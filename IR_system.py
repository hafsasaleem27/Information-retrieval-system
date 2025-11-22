import csv # import csv module
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

def collect_docs():

    # contains all docs (articles)
    docs = []

    with open('Articles.csv', 'r') as articles:
        file_reader = csv.reader(articles)
        for doc in file_reader:
            docs.append(doc)
    return docs

documents = collect_docs()
   
def get_content(documents):
    article_col_index = 0
    heading_col_index = 2
    article_content = []    # this will hold articles
    heading_content = []    # this will hold headings
    
    for doc in documents:
        article_content.append(doc[article_col_index])
        heading_content.append(doc[heading_col_index])
    return [article_content, heading_content]

def text_preprocess(text):
    tokens = word_tokenize(text)

    # create a list of stopwords in English
    stop_words = set(stopwords.words("english"))

    stopword_list = []
    for word in tokens:
        if word.casefold() not in stop_words:
            stopword_list.append(word)
    
    # perform lemmatization
    lemmatizer = WordNetLemmatizer()

    lemmatized_words = []
    for word in stopword_list:
        lemmatized_words.append(lemmatizer.lemmatize(word))
    
    return lemmatized_words

def build_inverted_index(content): # content is a list of strings
    inverted_index = defaultdict(list)

    for doc_id, text in enumerate(content):
        lemmatized_words = text_preprocess(text)

        unique_terms = set(lemmatized_words)

        for term in unique_terms:
            inverted_index[term].append(doc_id)

    return inverted_index

[article_content, heading_content] = get_content(documents)
article_index = build_inverted_index(article_content) # build an inverted index for articles' content
heading_index = build_inverted_index(heading_content) # build an inverted index for headings' content
print(heading_index)