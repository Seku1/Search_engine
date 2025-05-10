import json
import numpy as np
import scipy.sparse as sp


def compute_idf(X, vocab_size):
    """
    Calculates IDF values for each word in the vocabulary.

    IDF(w) = log(N/nw), where:
    - N is the total number of documents
    - nw is the number of documents containing word w
    """
    N = X.shape[0]  # number of documents
    df = np.bincount(X.nonzero()[1], minlength=vocab_size)  # document frequency for each word
    # Avoid division by zero by adding 1 to numerator and denominator
    idf = np.log(np.divide(N + 1, df + 1))
    return idf


def apply_idf_to_bow(X, idf):
    """
    Multiplies each element of the BoW matrix by the corresponding IDF value
    """
    # Convert idf to diagonal matrix
    idf_diag = sp.diags(idf, 0)
    # Multiply BoW matrix by IDF
    X_tfidf = X.dot(idf_diag)
    return X_tfidf


def main():
    # Load vocabulary
    print("Loading vocabulary...")
    try:
        with open("vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
    except FileNotFoundError:
        print("Error: vocab.json not found")
        print("Please run create_bag_of_words.py first")
        return

    # Load BoW matrix
    print("Loading BoW matrix...")
    try:
        X = sp.load_npz("bow_matrix.npz")
    except FileNotFoundError:
        print("Error: bow_matrix.npz not found")
        print("Please run create_bag_of_words.py first")
        return

    # Calculate IDF values
    print(f"Calculating IDF for {len(vocab)} words...")
    idf = compute_idf(X, len(vocab))

    # Save IDF values to file
    np.save("idf_values.npy", idf)
    print("Saved IDF values to idf_values.npy")

    # Apply IDF to BoW matrix
    print("Creating TF-IDF matrix...")
    X_tfidf = apply_idf_to_bow(X, idf)

    # Save transformed matrix
    sp.save_npz("tfidf_matrix.npz", X_tfidf)
    print("Saved TF-IDF matrix to tfidf_matrix.npz")

    print("\nProcessing complete. You can now run app.py to start the web server or use search_documents.py for command-line search.")


if __name__ == "__main__":
    main()