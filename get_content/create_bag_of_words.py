import json
import re
import numpy as np
import scipy.sparse as sp
from collections import Counter
from tqdm import tqdm


def create_vocabulary(documents):
    """
    Creates a vocabulary (dictionary) from processed documents
    """
    print("Creating vocabulary...")
    all_words = Counter()

    for doc in tqdm(documents):
        words = doc.split()
        all_words.update(words)

    # Filter words that appear at least 5 times
    vocab_words = {word for word, count in all_words.items() if count >= 5}

    # Create dictionary mapping words to indices
    vocab = {word: idx for idx, word in enumerate(sorted(vocab_words))}

    print(f"Vocabulary size: {len(vocab)} words")
    return vocab


def documents_to_bow(documents, vocab):
    """
    Converts documents to a sparse bag-of-words matrix
    """
    print("Creating Bag-of-Words matrix...")
    rows = []
    cols = []
    data = []

    for doc_idx, doc in enumerate(tqdm(documents)):
        words = doc.split()
        word_counts = Counter(words)

        for word, count in word_counts.items():
            if word in vocab:
                rows.append(doc_idx)
                cols.append(vocab[word])
                data.append(count)

    # Create sparse matrix
    X = sp.csr_matrix((data, (rows, cols)), shape=(len(documents), len(vocab)))

    print(f"Created BoW matrix with shape: {X.shape}")
    return X


def main():
    print("Starting Bag-of-Words creation process...")

    # Load preprocessed documents
    print("Loading processed documents...")
    with open("wiki_procesed.json", "r", encoding="utf-8") as f:
        processed_docs = json.load(f)

    print(f"Loaded {len(processed_docs)} processed documents")

    # Create vocabulary
    vocab = create_vocabulary(processed_docs)

    # Save vocabulary to file
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    print("Saved vocabulary to vocab.json")

    # Create bag-of-words matrix
    X_bow = documents_to_bow(processed_docs, vocab)

    # Save bag-of-words matrix
    sp.save_npz("bow_matrix.npz", X_bow)
    print("Saved Bag-of-Words matrix to bow_matrix.npz")

    print("\nProcessing complete. You can now run compute_tfidf.py to create the TF-IDF matrix.")


if __name__ == "__main__":
    main()