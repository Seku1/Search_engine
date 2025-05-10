import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from collections import Counter
import os


def process_query(query, vocab, stemmer=None, stop_words=None):
    """
    Transforms a query into a bag-of-words vector
    """
    # Optional query processing like documents
    if stemmer and stop_words:
        query = query.lower()
        tokens = query.split()
        tokens = [stemmer.stem(tok) for tok in tokens if tok not in stop_words and len(tok) > 2]
    else:
        tokens = query.lower().split()

    # Create query vector
    query_counter = Counter(tokens)
    q = np.zeros(len(vocab))

    for token, count in query_counter.items():
        if token in vocab:
            q[vocab[token]] = count

    return q


def find_similar_documents_svd(query_vector, Uk, Sk, Vtk, k=5):
    """
    Finds k documents most similar to the query using a reduced SVD space.

    Args:
        query_vector: The query vector in original space
        Uk: Left singular vectors matrix
        Sk: Singular values vector
        Vtk: Right singular vectors matrix
        k: Number of documents to return

    Returns:
        Indices of most similar documents and their similarity scores
    """
    # Project query to SVD space
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return [], []  # Empty query

    # Normalize query
    query_vector = query_vector / query_norm

    # Project query to the reduced space
    # q' = q * Vk * inv(Sk)
    query_svd = query_vector.dot(Vtk.T) / Sk

    # Compute similarities with documents in the reduced space
    # Documents in the reduced space are the rows of Uk * Sk
    doc_vectors_svd = Uk * Sk

    # Calculate cosine similarities
    # Since both query_svd and rows of doc_vectors_svd are normalized
    # in the reduced space, we just need the dot product
    similarities = doc_vectors_svd.dot(query_svd)

    # Sort documents by similarity (descending)
    most_similar_indices = np.argsort(similarities)[::-1][:k]

    return most_similar_indices, similarities[most_similar_indices]


def compute_svd(X_tfidf, k_dimensions):
    """
    Compute SVD for the TF-IDF matrix with specified dimensions

    Args:
        X_tfidf: The TF-IDF matrix
        k_dimensions: Number of dimensions for the reduced space

    Returns:
        Uk, Sk, Vtk: SVD matrices
    """
    print(f"Computing SVD with {k_dimensions} dimensions...")
    Uk, Sk, Vtk = svds(X_tfidf, k=k_dimensions)
    return Uk, Sk, Vtk


def save_svd_matrices(Uk, Sk, Vtk, svd_dir, dimensions):
    """
    Save the SVD matrices to the specified directory

    Args:
        Uk, Sk, Vtk: SVD matrices
        svd_dir: Directory to save the matrices
        dimensions: Number of dimensions (used in filenames)
    """
    # Create directory if it doesn't exist
    os.makedirs(svd_dir, exist_ok=True)

    # Save matrices with dimension in filename
    np.save(f"{svd_dir}/svd_Uk_{dimensions}.npy", Uk)
    np.save(f"{svd_dir}/svd_Sk_{dimensions}.npy", Sk)
    np.save(f"{svd_dir}/svd_Vtk_{dimensions}.npy", Vtk)

    # Also save a config file with the dimensions
    with open(f"{svd_dir}/svd_config.json", "w") as f:
        json.dump({"dimensions": dimensions}, f)

    print(f"SVD matrices with {dimensions} dimensions saved to {svd_dir}")


def load_svd_matrices(svd_dir, dimensions=None):
    """
    Load SVD matrices from the specified directory

    Args:
        svd_dir: Directory containing SVD matrices
        dimensions: Specific dimensions to load (if None, loads from config)

    Returns:
        Uk, Sk, Vtk: SVD matrices or None if not found
    """
    try:
        # If dimensions not specified, try to load from config
        if dimensions is None:
            try:
                with open(f"{svd_dir}/svd_config.json", "r") as f:
                    config = json.load(f)
                dimensions = config.get("dimensions", 100)  # Default to 100 if not found
            except FileNotFoundError:
                dimensions = 100  # Default if config not found

        # Load the matrices
        Uk = np.load(f"{svd_dir}/svd_Uk_{dimensions}.npy")
        Sk = np.load(f"{svd_dir}/svd_Sk_{dimensions}.npy")
        Vtk = np.load(f"{svd_dir}/svd_Vtk_{dimensions}.npy")
        print(f"Successfully loaded SVD matrices with {dimensions} dimensions from {svd_dir}")
        return Uk, Sk, Vtk, dimensions

    except FileNotFoundError as e:
        print(f"Error loading SVD matrices: {e}")
        return None, None, None, dimensions


def main():
    # Default SVD directory and dimensions
    svd_dir = "svd_matrices"
    default_dimensions = 100

    # Check for file availability
    try:
        # Load vocabulary
        print("Loading vocabulary...")
        with open("vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Load TF-IDF matrix
        print("Loading TF-IDF matrix...")
        X_tfidf = sp.load_npz("tfidf_matrix.npz")

        # Load original documents (to display results)
        print("Loading original documents...")
        with open("wiki_documents.json", "r", encoding="utf-8") as f:
            original_docs = json.load(f)

        # Check if SVD matrices already exist
        Uk, Sk, Vtk, current_dimensions = load_svd_matrices(svd_dir)

        # If matrices don't exist or user wants to regenerate
        regenerate = False
        if Uk is None:
            print("SVD matrices not found. Will compute them.")
            regenerate = True
        else:
            user_choice = input(f"SVD matrices with {current_dimensions} dimensions already exist. Regenerate? (y/n): ")
            if user_choice.lower() == 'y':
                regenerate = True

        if regenerate:
            # Ask for dimensions
            try:
                input_dimensions = input(f"Enter number of SVD dimensions [default {default_dimensions}]: ")
                k_dimensions = int(input_dimensions) if input_dimensions.strip() else default_dimensions
            except ValueError:
                k_dimensions = default_dimensions
                print(f"Invalid input. Using default: {default_dimensions} dimensions")

            # Compute and save SVD matrices
            print(f"Performing SVD on TF-IDF matrix with {k_dimensions} dimensions...")
            Uk, Sk, Vtk = compute_svd(X_tfidf, k_dimensions)
            save_svd_matrices(Uk, Sk, Vtk, svd_dir, k_dimensions)
            current_dimensions = k_dimensions

        print(f"Ready to search in {len(original_docs)} documents using SVD with {current_dimensions} dimensions.")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Make sure you first run compute_tfidf.py")
        return

    # User interface for search
    print("\n=== Document Search System (SVD) ===")
    while True:
        query = input("\nEnter query (or 'q' to quit): ")
        if query.lower() == 'q':
            break

        if not query.strip():
            print("Query is empty. Please try again.")
            continue

        try:
            k = int(input("How many documents to display? [default 5]: ") or "5")
        except ValueError:
            k = 5

        # Process query
        query_vector = process_query(query, vocab)

        # Check if query contains known words
        if np.sum(query_vector) == 0:
            print("No word from the query exists in the dictionary. Try another query.")
            continue

        # Search for similar documents
        similar_indices, scores = find_similar_documents_svd(query_vector, Uk, Sk, Vtk, k)

        # Display results
        if len(similar_indices) == 0:
            print("No matching documents found.")
        else:
            print(f"\nFound {len(similar_indices)} documents similar to query '{query}':")
            for i, (idx, score) in enumerate(zip(similar_indices, scores), 1):
                # Cut document preview for readability
                doc_preview = original_docs[idx][:200] + "..." if len(original_docs[idx]) > 200 else original_docs[idx]
                print(f"\n{i}. Document #{idx} (similarity: {score:.4f})")
                print(f"   {doc_preview}")


if __name__ == "__main__":
    main()