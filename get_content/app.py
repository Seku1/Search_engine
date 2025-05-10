import json
import numpy as np
import scipy.sparse as sp
from flask import Flask, render_template, request, jsonify
from collections import Counter
import os
import os.path

app = Flask(__name__, static_folder='static')

# Global variables to store loaded data
vocab = None
X_tfidf = None
X_normalized = None
original_docs = None
# SVD matrices
Uk = None
Sk = None
Vtk = None
# Store current SVD dimensions
current_svd_dimensions = None
# SVD directory
SVD_DIR = "svd_matrices"


def load_data():
    """Load all necessary data files for the search system"""
    global vocab, X_tfidf, X_normalized, original_docs, Uk, Sk, Vtk, current_svd_dimensions

    # Load vocabulary
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Load TF-IDF matrix
    X_tfidf = sp.load_npz("tfidf_matrix.npz")

    # Load normalized matrix if it exists, otherwise compute it
    if os.path.exists("normalized_tfidf_matrix.npz"):
        X_normalized = sp.load_npz("normalized_tfidf_matrix.npz")
    else:
        X_normalized = normalize_matrix(X_tfidf)
        sp.save_npz("normalized_tfidf_matrix.npz", X_normalized)

    # Load original documents
    with open("wiki_documents.json", "r", encoding="utf-8") as f:
        original_docs = json.load(f)

    # Load default SVD matrices if they exist
    try:
        # Try to load from config first to get dimensions
        try:
            with open(f"{SVD_DIR}/svd_config.json", "r") as f:
                config = json.load(f)
            current_svd_dimensions = config.get("dimensions", 100)
        except FileNotFoundError:
            current_svd_dimensions = 100  # Default if config not found

        # Load matrices with the dimensions from config
        Uk = np.load(f"{SVD_DIR}/svd_Uk_{current_svd_dimensions}.npy")
        Sk = np.load(f"{SVD_DIR}/svd_Sk_{current_svd_dimensions}.npy")
        Vtk = np.load(f"{SVD_DIR}/svd_Vtk_{current_svd_dimensions}.npy")
        print(f"SVD matrices loaded successfully with {current_svd_dimensions} dimensions")
    except FileNotFoundError:
        print("SVD matrices not found. SVD search will not be available until computed.")

    return True


def process_query(query, vocab):
    """
    Transforms a query into a bag-of-words vector
    """
    tokens = query.lower().split()

    # Create query vector
    query_counter = Counter(tokens)
    q = np.zeros(len(vocab))

    for token, count in query_counter.items():
        if token in vocab:
            q[vocab[token]] = count

    return q


def normalize_matrix(X):
    """
    Normalizes each row of matrix X to have length 1
    """
    # Calculate norm of each row
    row_norms = np.sqrt((X.power(2)).sum(axis=1).A1)

    # Avoid division by zero
    row_norms[row_norms == 0] = 1.0

    # Create diagonal matrix with inverse norms
    inv_norms = sp.diags(1.0 / row_norms, 0)

    # Normalize each row
    X_normalized = inv_norms.dot(X)

    return X_normalized


def find_similar_documents(query_vector, X_tfidf, k=5):
    """
    Finds k documents most similar to the query using cosine similarity
    """
    # Calculate query norm
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return [], []  # No query

    # Normalize query vector
    query_vector = query_vector / query_norm

    # Prepare query vector for matrix multiplication
    query_vector = query_vector.reshape(-1, 1)  # as a column vector

    # Calculate dot product of query with each document
    dot_products = X_tfidf.dot(query_vector).flatten()

    # Calculate document norms
    doc_norms = np.sqrt((X_tfidf.power(2)).sum(axis=1).A1)

    # Avoid division by zero
    doc_norms[doc_norms == 0] = 1.0

    # Calculate cosine similarity
    similarities = dot_products / doc_norms

    # Sort documents by similarity (descending)
    most_similar_indices = np.argsort(similarities)[::-1][:k]

    return most_similar_indices, similarities[most_similar_indices]


def find_similar_documents_abs_cosine(query_vector, X_normalized, k=5):
    """
    Finds k documents most similar to the query using
    absolute value of cosine similarity
    """
    # Normalize query vector
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return [], []  # No query

    # Normalize query vector
    query_vector = query_vector / query_norm

    # Prepare query vector for matrix multiplication
    query_vector = query_vector.reshape(-1, 1)  # as a column vector

    # Calculate dot product of query with each document
    dot_products = X_normalized.dot(query_vector).flatten()

    # Calculate absolute value of cosine similarity
    abs_similarities = np.abs(dot_products)

    # Sort documents by similarity (descending)
    most_similar_indices = np.argsort(abs_similarities)[::-1][:k]

    return most_similar_indices, abs_similarities[most_similar_indices]


def find_similar_documents_svd(query_vector, Uk, Sk, Vtk, k=5):
    """
    Finds k documents most similar to the query using a reduced SVD space
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
    similarities = doc_vectors_svd.dot(query_svd)

    # Sort documents by similarity (descending)
    most_similar_indices = np.argsort(similarities)[::-1][:k]

    return most_similar_indices, similarities[most_similar_indices]


def compute_svd(X_tfidf, k_dimensions):
    """
    Compute SVD for the TF-IDF matrix with the specified dimensions

    Args:
        X_tfidf: The TF-IDF matrix
        k_dimensions: Number of dimensions for reduced space

    Returns:
        Uk, Sk, Vtk: SVD matrices
    """
    from scipy.sparse.linalg import svds
    print(f"Computing SVD with {k_dimensions} dimensions...")
    return svds(X_tfidf, k=k_dimensions)


def save_svd_matrices(Uk, Sk, Vtk, dimensions):
    """
    Save SVD matrices to the SVD directory

    Args:
        Uk, Sk, Vtk: SVD matrices
        dimensions: Number of dimensions
    """
    global current_svd_dimensions

    # Create SVD directory if it doesn't exist
    os.makedirs(SVD_DIR, exist_ok=True)

    # Save matrices with dimension in filename
    np.save(f"{SVD_DIR}/svd_Uk_{dimensions}.npy", Uk)
    np.save(f"{SVD_DIR}/svd_Sk_{dimensions}.npy", Sk)
    np.save(f"{SVD_DIR}/svd_Vtk_{dimensions}.npy", Vtk)

    # Save config with current dimensions
    with open(f"{SVD_DIR}/svd_config.json", "w") as f:
        json.dump({"dimensions": dimensions}, f)

    # Update current dimensions
    current_svd_dimensions = dimensions
    print(f"SVD matrices with {dimensions} dimensions saved to {SVD_DIR}")


def list_available_svd_dimensions():
    """
    List all available SVD dimensions by scanning the SVD directory

    Returns:
        List of available dimensions
    """
    dimensions = []

    # Create directory if it doesn't exist
    os.makedirs(SVD_DIR, exist_ok=True)

    # Scan for Uk files which follow the pattern svd_Uk_<dimensions>.npy
    for filename in os.listdir(SVD_DIR):
        if filename.startswith("svd_Uk_") and filename.endswith(".npy"):
            # Extract dimensions from filename
            try:
                dim = int(filename[7:-4])  # Remove "svd_Uk_" and ".npy"
                dimensions.append(dim)
            except ValueError:
                continue

    return sorted(dimensions)


@app.route('/')
def index():
    """Render the main search page"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    global Uk, Sk, Vtk, current_svd_dimensions

    data = request.get_json()
    query = data.get('query', '')
    k = int(data.get('k', 5))
    search_type = data.get('search_type', 'regular')

    # For SVD search, get dimensions parameter if provided
    svd_dimensions = data.get('svd_dimensions', current_svd_dimensions)

    if not query.strip():
        return jsonify({"error": "Query is empty"}), 400

    # Process query
    query_vector = process_query(query, vocab)

    # Check if query contains known words
    if np.sum(query_vector) == 0:
        return jsonify({"error": "No word from the query exists in the dictionary"}), 400

    # Search for similar documents based on search type
    if search_type == 'absolute':
        similar_indices, scores = find_similar_documents_abs_cosine(query_vector, X_normalized, k)
    elif search_type == 'svd':
        # Check if we need to load different SVD matrices
        if svd_dimensions != current_svd_dimensions:
            try:
                Uk = np.load(f"{SVD_DIR}/svd_Uk_{svd_dimensions}.npy")
                Sk = np.load(f"{SVD_DIR}/svd_Sk_{svd_dimensions}.npy")
                Vtk = np.load(f"{SVD_DIR}/svd_Vtk_{svd_dimensions}.npy")
                current_svd_dimensions = svd_dimensions
                print(f"Loaded SVD matrices with {svd_dimensions} dimensions")
            except FileNotFoundError:
                return jsonify({"error": f"SVD matrices with {svd_dimensions} dimensions not found."}), 400

        if Uk is None or Sk is None or Vtk is None:
            return jsonify({"error": "SVD matrices not loaded. Compute SVD matrices first."}), 400

        similar_indices, scores = find_similar_documents_svd(query_vector, Uk, Sk, Vtk, k)
    else:  # regular cosine
        similar_indices, scores = find_similar_documents(query_vector, X_tfidf, k)

    # Prepare results
    results = []
    for idx, score in zip(similar_indices, scores):
        doc_preview = original_docs[idx][:500] + "..." if len(original_docs[idx]) > 500 else original_docs[idx]
        results.append({
            "index": int(idx),
            "score": float(score),
            "preview": doc_preview
        })

    return jsonify({
        "results": results,
        "search_type": search_type,
        "svd_dimensions": current_svd_dimensions if search_type == 'svd' else None
    })


@app.route('/document/<int:doc_id>')
def get_document(doc_id):
    """Return a full document by ID"""
    if doc_id < 0 or doc_id >= len(original_docs):
        return jsonify({"error": "Document ID out of range"}), 404

    return jsonify({"document": original_docs[doc_id]})


@app.route('/svd-dimensions')
def get_svd_dimensions():
    """Return available SVD dimensions"""
    dimensions = list_available_svd_dimensions()
    return jsonify({
        "available_dimensions": dimensions,
        "current_dimensions": current_svd_dimensions
    })


@app.route('/compute-svd', methods=['POST'])
def compute_svd_endpoint():
    """Compute and save SVD matrices with the specified dimensions"""
    global Uk, Sk, Vtk, current_svd_dimensions

    data = request.get_json()
    dimensions = int(data.get('dimensions', 100))

    if dimensions < 1:
        return jsonify({"error": "Dimensions must be at least 1"}), 400

    try:
        # Compute SVD
        Uk, Sk, Vtk = compute_svd(X_tfidf, dimensions)

        # Save matrices
        save_svd_matrices(Uk, Sk, Vtk, dimensions)

        return jsonify({
            "success": True,
            "message": f"SVD matrices with {dimensions} dimensions computed and saved successfully",
            "dimensions": dimensions
        })
    except Exception as e:
        return jsonify({"error": f"Error computing SVD: {str(e)}"}), 500


if __name__ == "__main__":
    # Load all data before starting the server
    print("Loading data...")
    load_data()
    print(f"Ready for searching in {len(original_docs)} documents")

    # Print available SVD dimensions
    dimensions = list_available_svd_dimensions()
    if dimensions:
        print(f"Available SVD dimensions: {dimensions}")
        print(f"Current SVD dimensions: {current_svd_dimensions}")
    else:
        print("No SVD matrices found. Use the /compute-svd endpoint to compute them.")

    # Start the server
    app.run(debug=True)