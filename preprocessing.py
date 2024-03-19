import io
import string
import pandas as pd
import spacy
from gensim.parsing import PorterStemmer


def tokenize(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return [token.text for token in doc]


def lowercase(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop]


def stem(text):
    ps = PorterStemmer()
    return [ps.stem(word) for word in text]


def lemmatize(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return [token.lemma_ for token in doc]


def remove_whitespace(text):
    return " ".join(text.split())


def vectorisation(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return doc.vector


def do_preprocessing(text):
    text = lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = stem(text)
    text = lemmatize(text)
    text = remove_whitespace(text)
    return text
