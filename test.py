import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
#nltk.download('punkt')
#nltk.download('stopwords')
import re

# Load the text data
with open('female.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Tokenize the text into words
words = word_tokenize(data)

# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Calculate the average sentence length
avg_sent_length = sum(len(sent) for sent in sentences) / len(sentences)

# Count the frequency of each word
word_freq = Counter(words)

# Count the number of stopwords
stop_words = set(stopwords.words('english'))
num_stopwords = sum(freq for word, freq in word_freq.items() if word in stop_words)

# Count the number of punctuation marks
num_punctuation = sum(freq for word, freq in word_freq.items() if word in string.punctuation)

# Extract citation practices
citations = re.findall(r'\[\d+\]', data)
num_citations = len(citations)

# Print the results
print(f'Average sentence length: {avg_sent_length}')
print(f'Number of stopwords: {num_stopwords}')
print(f'Number of punctuation marks: {num_punctuation}')
print(f'Number of citations: {num_citations}')

