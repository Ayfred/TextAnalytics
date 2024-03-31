import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv("./dataset/features.csv")

# Initialize an empty list to store textual columns
textual_columns = []

X_text = data['most_common_words']

y = data['gender']
gender_mapping = {'female': 0, 'male': 1}
y = y.map(gender_mapping)

# Transform textual data to numerical data using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_text_vectorized = tfidf_vectorizer.fit_transform(X_text) 

# 2. Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Combine numerical features and target variable
preprocessed_data = pd.DataFrame(X_text_vectorized.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
preprocessed_data['gender'] = y_encoded

data = data.drop('most_common_words', axis=1)
data = data.drop('gender', axis=1)
preprocessed_data = pd.concat([data, preprocessed_data], axis=1)

# relocate gender to first column
cols = preprocessed_data.columns.tolist()
cols = cols[-1:] + cols[:-1]
preprocessed_data = preprocessed_data[cols]

# Save preprocessed data to a new CSV file
preprocessed_data.to_csv("./dataset/preprocessed_data.csv", index=False)

print("Preprocessing finished")