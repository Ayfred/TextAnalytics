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
    if isinstance(text, str):
        doc = nlp(text)
        return [token.lemma_ for token in doc]
    elif isinstance(text, list):
        lemmatized_texts = []
        for t in text:
            doc = nlp(t)
            lemmatized_texts.append([token.lemma_ for token in doc])
        return lemmatized_texts
    else:
        raise ValueError("Input should be a string or a list of strings.")


def remove_whitespace(text):
    if isinstance(text, str):
        return " ".join(text.split())
    elif isinstance(text, list):
        return [" ".join(str(t).split()) for t in text]
    else:
        raise ValueError("Input should be a string or a list of strings.")


def vectorisation(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return doc.vector

def average_sentence_length(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len(doc) / len(list(doc.sents))

def average_word_length(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len(doc) / len(list(doc))

def count_punctuation(text):
    return sum([1 for char in text if char in string.punctuation])


def most_common_words(text, n=10):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    word_freq = pd.Series(words).value_counts()
    return word_freq.head(n)

def lexical_diversity(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len(set([token.text for token in doc])) / len([token.text for token in doc])

def semantic(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return doc.vector

def word_count(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len(doc)

def count_verbs(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "VERB"])

def count_nouns(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "NOUN"])

def count_adjectives(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "ADJ"])

def count_adverbs(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "ADV"])

def count_pronouns(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "PRON"])

def count_conjunctions(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "CCONJ"])

def count_prepositions(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "ADP"])

def count_determiners(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "DET"])

def count_numbers(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "NUM"])

def count_foreign_words(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.is_oov])

def count_wh_words(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB"])

def sentiment_score(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return doc._.polarity

def count_slang(text):
    slang_words = [
        "lit", "fam", "yeet", "bruh", "flex", "bae", "GOAT", "on fleek", "squad", "thirsty",
        "turnt", "woke", "chill", "savage", "thicc", "stan", "sus", "vibe", "basic", "AF",
        "TBH", "YOLO", "swerve", "smh", "LOL", "WTF", "OMG", "BTW", "ICYMI", "IDK", "TL;DR",
        "IMO", "IMHO", "ROFL", "BRB", "JK", "FTW", "LMAO", "TMI", "OOTD", "FOMO", "AMA", "DM",
        "FF", "ICYMI", "NSFW", "TLDR", "TBT", "TIL", "IMO", "TLDR", "GTFO", "ICYMI", "OOTD",
        "IMO", "IRL", "WBU", "BFF", "BTFO", "WYD", "MFW", "MRW", "AMA", "Dank", "Salty",
        "Lit AF", "Shook", "Gucci", "Slay", "Extra", "Hundo P", "Sis", "Mood", "No cap",
        "Bless up", "Woke", "Bop", "Wig", "Flex", "Snatched", "Swag", "Hype", "Glow up",
        "Sip tea", "Keep it 100", "G.O.A.T", "Throw shade", "Tea", "Sus", "Ratchet", "Stan",
        "Hangry", "Spill the tea", "Thirst trap", "Ship", "Tea", "Hundo", "Ship", "Deadass",
        "Troll", "Ship", "Troll", "Feels", "Simp", "Blessed", "Salty", "Cray", "Wig", "Turn up",
        "Highkey", "Lowkey", "Finsta", "Thicc", "Swerve", "Bop", "Curve", "Salty", "Savage",
        "Thirsty", "Troll", "Yas", "Yeet"
    ]
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text.lower())  # Convert text to lowercase for case-insensitive matching
    return len([token.text for token in doc if token.text.lower() in slang_words])


def pronoun_density(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "PRON"]) / len([token.text for token in doc])

def delete_backslash_n(text):
    return text.replace('\n', '')


def do_preprocessing(text):
    text = delete_backslash_n(text)

    print("Average sentence length: ", average_sentence_length(text))
    print("Average word length: ", average_word_length(text))
    print("Count punctuation: ", count_punctuation(text))
    print("Most common words: ", most_common_words(text))
    print("Lexical diversity: ", lexical_diversity(text))
    print("Word count: ", word_count(text))
    print("Count verbs: ", count_verbs(text))
    print("Count nouns: ", count_nouns(text))
    print("Count adjectives: ", count_adjectives(text))
    print("Count adverbs: ", count_adverbs(text))
    print("Count pronouns: ", count_pronouns(text))
    print("Count conjunctions: ", count_conjunctions(text))
    print("Count prepositions: ", count_prepositions(text))
    print("Count determiners: ", count_determiners(text))
    print("Count numbers: ", count_numbers(text))
    print("Count foreign words: ", count_foreign_words(text))
    print("Count wh words: ", count_wh_words(text))
    #print("Average sentiment score: ", sentiment_score(text))
    print("Count slang: ", count_slang(text))
    print("Pronoun density: ", pronoun_density(text))


    #text = lowercase(text)
    #text = remove_punctuation(text)
    #text = remove_stopwords(text)
    # text = stem(text)
    # text = lemmatize(text)
    #text = remove_whitespace(text)
    return text
