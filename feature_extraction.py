from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


def bag_of_words(text):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text)
    vocabulary = vectorizer.get_feature_names_out()
    bow_matrix_dense = X.toarray()
    return bow_matrix_dense, vocabulary

def tf_idf(text):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    vocabulary = vectorizer.get_feature_names_out()
    tf_idf_matrix_dense = X.toarray()
    return tf_idf_matrix_dense, vocabulary

def word2vec(text):
    model = Word2Vec(text, min_count=1)
    return model

def word_embeddings(text):
    tokenized_data = [word_tokenize(sentence.lower()) for sentence in text]

    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

    # Get the word embeddings
    word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
