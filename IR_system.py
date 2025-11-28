import csv # import csv module
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import math

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

def calculate_tf(words):
    total_terms = len(words)
    terms = defaultdict(int)
    for term in words:
        terms[term] += 1
    for term in terms:
        terms[term] = terms[term] / total_terms
    return terms

def calculate_idf(content):
    N = len(content)
    dfs = defaultdict(int)

    docs = []
    for text in content:
        words = text_preprocess(text)
        docs.append(words)
        unique_words = set(words)
        for term in unique_words:
            dfs[term] += 1

    idf = {}
    for word, df in dfs.items():
        idf[word] = math.log(N / df, 10)
    return idf, docs
        

def build_inverted_index(content): # content is a list of strings
    inverted_index = defaultdict(list)
    idf, docs = calculate_idf(content)

    for doc_id, words in enumerate(docs):
        tfs = calculate_tf(words)
        
        for term, tf in tfs.items():
            tf_idf = tf * idf[term]
            inverted_index[term].append((doc_id, tf_idf))

    return inverted_index

[article_content, heading_content] = get_content(documents)
article_index = build_inverted_index(article_content) # build an inverted index for articles' content
heading_index = build_inverted_index(heading_content) # build an inverted index for headings' content
print(heading_index)

# query processing
query = input("Enter a query: ")

def calculate_idf_query(content, query): # query is a string # content is a list of strings
    N = len(content)
    dfs = defaultdict(int)
    
    query_words = set(text_preprocess(query))

    for doc in content:
        doc_words = set(text_preprocess(doc))

        for word in query_words:
            if word in doc_words:
                dfs[word] += 1
   
    idf = {}
    for word in query_words:
        df = dfs[word] if dfs[word] > 0 else 1
        idf[word] = math.log(N / df, 10)
    return idf

query_words = text_preprocess(query)
terms = calculate_tf(query_words)