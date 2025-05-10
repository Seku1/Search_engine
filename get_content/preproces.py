import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm


def download_nltk_resources():
    """Download required NLTK resources if not already downloaded"""
    try:
        # Check if stopwords are available
        stopwords.words("english")
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords")

    # Make sure stemmer is available (should be included with NLTK)
    try:
        SnowballStemmer("english")
    except:
        print("Error with stemmer, NLTK might need to be updated")


def preprocess(text):
    """
    Preprocesses text by:
    1. Converting to lowercase
    2. Removing non-alphabetic characters
    3. Removing stopwords
    4. Stemming words
    """
    # Get stopwords and stemmer
    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")

    # Convert to lowercase
    text = text.lower()

    # Remove non-alphabetic characters (keeps spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Split into tokens
    tokens = text.split()

    # Remove stopwords and short words, and stem
    cleaned = [
        stemmer.stem(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 2
    ]

    # Join tokens back into a string
    return " ".join(cleaned)


def main():
    print("Starting document preprocessing...")

    # Download required NLTK resources
    download_nltk_resources()

    # Load original documents
    print("Loading original documents...")
    try:
        with open("wiki_documents.json", "r", encoding="utf-8") as f:
            articles = json.load(f)
        print(f"Loaded {len(articles)} documents")
    except FileNotFoundError:
        print("Error: wiki_documents.json not found")
        print("Please run get_wiki.py first to download Wikipedia articles")
        return

    # Preprocess documents
    print("Preprocessing documents...")
    processed_docs = []
    for doc in tqdm(articles):
        processed_docs.append(preprocess(doc))

    # Save processed documents
    print("Saving processed documents...")
    with open("wiki_procesed.json", "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False)

    print(f"Successfully processed and saved {len(processed_docs)} documents")
    print("Next step: Run create_bag_of_words.py to create the vocabulary and BoW matrix")


if __name__ == "__main__":
    main()