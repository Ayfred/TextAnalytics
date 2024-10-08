import string
import pandas as pd
import spacy
from gensim.parsing import PorterStemmer
from spacytextblob.spacytextblob import SpacyTextBlob
from collections import Counter
import re
import numpy as np

nlp = spacy.load('en_core_web_sm')

def tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop]

def stem(text):
    ps = PorterStemmer()
    return [ps.stem(word) for word in text]

def lemmatize(text):
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
    doc = nlp(text)
    return doc.vector

def average_sentence_length(text):
    doc = nlp(text)
    return len(doc) / len(list(doc.sents))

def average_word_length(text):
    doc = nlp(text)
    return len(doc) / len(list(doc))

def total_count_punctuation(text):
    return sum([1 for char in text if char in string.punctuation])

def count_particular_punctuation(text):
    punctuations = [".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "|", "_", "+", "=", "*", "&", "^", "%", "$", "#", "@", "~", "`"]
    count = {}
    for p in punctuations:
        count[p] = text.count(p)
    return count

def most_common_words(text, n=10):
    doc = nlp(text)
    words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return words[:n]  # Return the list of words with the specified number (n)

def lexical_diversity(text):
    doc = nlp(text)
    return len(set([token.text for token in doc])) / len([token.text for token in doc])

def semantic(text):
    doc = nlp(text)
    return doc.vector

def word_count(text):
    doc = nlp(text)
    return len(doc)

def count_pos(text):
    doc = nlp(text)
    pos_counts = {}
    pos_to_count = ["VERB", "NOUN", "ADJ", "ADV", "PRON", "CCONJ", "ADP", "DET", "NUM", "X", "INTJ", "SYM", "PART", "SPACE", "PUNCT", "SCONJ", "PROPN", "AUX", "CONJ"]

    for pos_tag in pos_to_count:
        pos_counts[pos_tag] = len([token.text for token in doc if token.pos_ == pos_tag])

    return pos_counts


def count_foreign_words(text):
    doc = nlp(text)
    return len([token.text for token in doc if token.is_oov])

def count_wh_words(text):
    doc = nlp(text)
    return len([token.text for token in doc if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB"])

def get_sentiment_score(text):
    doc = nlp(text)
    
    text_blob = SpacyTextBlob(nlp)
    doc = text_blob(doc)
    
    sentiment_score = doc._.polarity
    return sentiment_score

def get_subjectivity_score(text):
    doc = nlp(text)
    
    text_blob = SpacyTextBlob(nlp)
    doc = text_blob(doc)
    
    subjectivity_score = doc._.subjectivity
    return subjectivity_score

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
    doc = nlp(text.lower())  # Convert text to lowercase for case-insensitive matching
    return len([token.text for token in doc if token.text.lower() in slang_words])


def pronoun_richness(text):
    doc = nlp(text)
    return len([token.text for token in doc if token.pos_ == "PRON"]) / len([token.text for token in doc])

def pronoun_density(text, pronoun):
    doc = nlp(text)
    return len([token.text for token in doc if token.text == pronoun]) / len(doc)

def calculate_pronoun_densities(text):
    pronouns = ["I", "you", "he", "she", "they", "it", "we", "them", "my", "your", "his", "her", "their", "our", "its", "us", "me", "him", "her", "them", "myself", "yourself", "himself", "herself", "themselves", "ourselves"]
    pronoun_densities = {}
    for pronoun in pronouns:
        pronoun_densities[pronoun] = pronoun_density(text, pronoun)
    return pronoun_densities

def delete_backslash_n(text):
    return text.replace('\n', '')

def pos_distribution(text):
    doc = nlp(text)
    pos_counts = Counter()
    
    for token in doc:
        pos_counts[token.pos_] += 1
    
    total_tokens = len(doc)
    pos_distribution = {}
    
    all_pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB']
    
    for pos_tag in all_pos_tags:
        pos_distribution[pos_tag] = pos_counts[pos_tag] / total_tokens if pos_counts[pos_tag] else 0
    
    return pos_distribution


def pos_statistics(text):
    doc = nlp(text)
    pos_counts = Counter()
    
    for token in doc:
        pos_counts[token.pos_] += 1
    
    counts_array = np.array(list(pos_counts.values()))
    
    mean_count = np.mean(counts_array)
    median_count = np.median(counts_array)
    std_deviation = np.std(counts_array)
    max_count = np.max(counts_array)
    min_count = np.min(counts_array)
    
    return {
        'mean_count': mean_count,
        'median_count': median_count,
        'std_deviation': std_deviation,
        'max_count': max_count,
        'min_count': min_count
    }

import os

def extract_features(text):
    features = {}
    features['word_count'] = word_count(text)
    features['average_sentence_length'] = average_sentence_length(text)
    features['average_word_length'] = average_word_length(text)
    features['total_count_punctuation'] = total_count_punctuation(text)
    features['most_common_words'] = most_common_words(text, n= 1)[0]
    features['lexical_diversity'] = lexical_diversity(text)
    features['count_foreign_words'] = count_foreign_words(text)
    features['count_wh_words'] = count_wh_words(text)
    features['sentiment_score'] = get_sentiment_score(text)
    features['subjectivity_score'] = get_subjectivity_score(text)
    features['count_slang'] = count_slang(text)
    features['pronoun_richness'] = pronoun_richness(text)

    pos = count_pos(text)
    for pos_tag, count in pos.items():
        features[f'count_{pos_tag}'] = count

    particular_punctuations = count_particular_punctuation(text)
    for punctuation, count in particular_punctuations.items():
        punctuation_cleaned = (punctuation
                               .replace(';', 'semi_colon')
                               .replace(':', 'colon')
                               .replace('!', 'exclamation')
                               .replace('?', 'question')
                               .replace(',', 'comma')
                               .replace('.', 'period')
                               .replace('-', 'dash'))
        features[f'count_{punctuation_cleaned}'] = count

    pos_distribution_values = pos_distribution(text)
    for pos_tag, distribution in pos_distribution_values.items():
        features[f'pos_distribution_{pos_tag}'] = distribution

    pronoun_density_values = calculate_pronoun_densities(text)
    for pronoun, density in pronoun_density_values.items():
        features[f'pronoun_density_{pronoun}'] = density


    pos_statistics_values = pos_statistics(text)
    for stat, value in pos_statistics_values.items():
        features[f'pos_statistics_{stat}'] = value

    return features

# Function to determine target value based on file name
def get_target(file_name):
    if re.match(r'^male', file_name):
        return 'male'
    elif re.match(r'^female', file_name):
        return 'female'
    else:
        return None 

# Function to process each file
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        features = extract_features(text)
        target = get_target(os.path.basename(file_path))
        return {**features, 'gender': target}

# Main function to process all files in the folder
def process_folder(folder_path):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            data.append(process_file(file_path))
    return data

# Export data to CSV
def export_to_csv(data, csv_filename):
    df = pd.DataFrame(data)
    
    cols = df.columns.tolist()
    if 'gender' in cols:
        cols.insert(0, cols.pop(cols.index('gender')))
    
    df = df[cols]  # Reorder DataFrame columns
    df.to_csv(csv_filename, index=False)

def main():
    folder_path = './data'
    csv_filename = './dataset/features.csv'
    data = process_folder(folder_path)
    export_to_csv(data, csv_filename)
    print("CSV export successful.")

if __name__ == "__main__":
    main()
