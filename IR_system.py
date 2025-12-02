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
    # tokenize text
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

def calculate_idf(content): # content is a list of strings
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
            inverted_index[term].append((doc_id, tf_idf)) # list of tuples

    return inverted_index

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

# Collect candidate documents
def collect_candidate_docs(query, inverted_index):
    query_words = text_preprocess(query)
    candidate_doc_ids = set()

    for word in query_words:
        if word in inverted_index:
            candidate_doc_ids |= {doc_id for doc_id, _ in inverted_index[word]}

    return candidate_doc_ids

def calculate_query_vector(query_words, idf):
    query_vector = {}
    terms = calculate_tf(query_words)
    for term in terms:
        query_vector[term] = terms[term] * idf[term]
    return query_vector

def cal_doc_vectors(candidate_doc_ids, query_words, inverted_index):
    document_vectors = {doc_id: {} for doc_id in candidate_doc_ids}
    # candidate_doc_ids is a set
    # query_words is a list of strings
    # inverted index is a dictionary with value being list of tuples
    for word in query_words:
        if word in inverted_index:
            for tup in inverted_index[word]:
                if tup[0] in candidate_doc_ids:
                    document_vectors[tup[0]][word] = tup[1]
                    # place tf-idf value here after matching candidate_id
    return document_vectors

def calculate_dot_product(query_vector, document_vector):
    dot_product = 0
    for term in query_vector:
        if term in document_vector:
            dot_product += query_vector[term] * document_vector[term]
    return dot_product

def compute_magnitude(vector):
    magnitude = 0
    for term in vector:
        magnitude += vector[term] ** 2
    magnitude = math.sqrt(magnitude)
    return magnitude

def calculate_cosine_sim(query_vector, document_vector):
    product_val = calculate_dot_product(query_vector, document_vector)
    query_magnitude = compute_magnitude(query_vector)
    document_magnitude = compute_magnitude(document_vector)
    return (product_val / (query_magnitude * document_magnitude))

def merge_content(first_content, second_content):
    content_list = []
    for element1, element2 in zip(first_content, second_content):
        content_list.append(element1 + " " + element2)
    return content_list

def calculate_cosine_similarities(query_vector, document_vectors):
    cosine_similarities = {}
    for doc_id, vector in document_vectors.items():
        # calculate cosine similarity
        cosine_similarities[doc_id] = calculate_cosine_sim(query_vector, vector)
    return cosine_similarities

def rank_results(cosine_similarities):
    ranked_docs = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def display_documents(ranked_results, article_content, heading_content):
    for doc_id, score in ranked_results:
        heading = heading_content[doc_id]
        content = article_content[doc_id]

        print(f"Rank: {doc_id} | Score: {score:.4f}")
        print("Heading:", heading)
        print("Content:", content)
        print("-" * 50)

# collect docs in a list
documents = collect_docs()
[article_content, heading_content] = get_content(documents)

# merge article_content and heading_content into one list
merged_list = merge_content(heading_content, article_content)

inverted_index = build_inverted_index(merged_list) # build an inverted index

# query processing
query = input("Enter a query: ")
query_words = text_preprocess(query)

# calculate candidate document ids
candidate_doc_ids = collect_candidate_docs(query, inverted_index)

# calculate query idf
query_idf = calculate_idf_query(merged_list, query)

# calculate query vector
query_vector = calculate_query_vector(query_words, query_idf)

# calculate document vectors for candidate documents
document_vectors = cal_doc_vectors(candidate_doc_ids, query_words, inverted_index)

# calculate cosine similarities for each query/doc pair
cos_similarities = calculate_cosine_similarities(query_vector, document_vectors)

# rank results
ranked_results = rank_results(cos_similarities)

# print results
display_documents(ranked_results, article_content, heading_content)

# print(ranked_results)