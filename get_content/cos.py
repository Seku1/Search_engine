import json
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # zostaw litery, cyfry, myślniki
    tokens = text.split()
    cleaned = [
        stemmer.stem(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 2
    ]
    return " ".join(cleaned)

# Zastosuj do wszystkich dokumentów

with open("wiki_documents.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

processed_docs = [preprocess(doc) for doc in articles]

with open("wiki_procesed.json", "w", encoding="utf-8") as f:
    json.dump(processed_docs, f, ensure_ascii=False, indent=2)

print(f"\nZapisano {len(processed_docs)} dokumentów.")

print(f"Liczba artykułów: {len(articles)}")