import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data - dummy SQL codes
data = {
    'ID': [1, 2, 3, 4, 5],
    'SQL_Code': [
        "SELECT * FROM table1 WHERE condition1;",
        "SELECT column1, column2 FROM table2 WHERE condition2;",
        "SELECT column1 FROM table1 WHERE condition1 AND condition2;",
        "SELECT * FROM table3 WHERE condition3;",
        "SELECT column1, column2 FROM table1 WHERE condition4;"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm") # Integrating spaCy with SQL Server Machine Learning Services (MLS)


# Function to preprocess SQL codes
def preprocess_sql(sql_code):
    # Tokenize SQL code using spaCy
    doc = nlp(sql_code)
    # Remove stop words and punctuation, and lemmatize tokens
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Preprocess SQL codes
df['Preprocessed_SQL'] = df['SQL_Code'].apply(preprocess_sql)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Preprocessed_SQL'])

# Compute cosine similarity between text vectors
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Threshold for cosine similarity
threshold = 0.5  # Adjust as needed

# Find matching phrases based on cosine similarity
matching_phrases = {}
for i, row in df.iterrows():
    matching_phrases[row['SQL_Code']] = set()  # Initialize the set
    for j, similarity in enumerate(cosine_similarities[i]):
        if i != j and similarity > threshold:
            matching_phrases[row['SQL_Code']].add(df.at[j, 'ID'])

# Calculate the length of the associated ID set
df['Associated_ID_Length'] = df['SQL_Code'].apply(lambda x: len(matching_phrases[x]))

# Print only phrases associated with more than one ID
for index, row in df.iterrows():
    if row['Associated_ID_Length'] > 1:
        print(f"Phrase: {row['SQL_Code']}, IDs: {matching_phrases[row['SQL_Code']]}")
