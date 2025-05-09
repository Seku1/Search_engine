import json
import numpy as np
import scipy.sparse as sp
from collections import Counter


def process_query(query, vocab, stemmer=None, stop_words=None):
    """
    Przekształca zapytanie w wektor bag-of-words
    """
    # Opcjonalne przetwarzanie zapytania jak dokumentów
    if stemmer and stop_words:
        query = query.lower()
        tokens = query.split()
        tokens = [stemmer.stem(tok) for tok in tokens if tok not in stop_words and len(tok) > 2]
    else:
        tokens = query.lower().split()

    # Tworzenie wektora zapytania
    query_counter = Counter(tokens)
    q = np.zeros(len(vocab))

    for token, count in query_counter.items():
        if token in vocab:
            q[vocab[token]] = count

    return q


def normalize_matrix(X):
    """
    Normalizuje każdy wiersz macierzy X, aby miał długość 1
    """
    # Obliczenie normy każdego wiersza
    row_norms = np.sqrt((X.power(2)).sum(axis=1).A1)

    # Unikanie dzielenia przez zero
    row_norms[row_norms == 0] = 1.0

    # Tworzenie diagonalnej macierzy z odwrotnościami norm
    inv_norms = sp.diags(1.0 / row_norms, 0)

    # Normalizacja każdego wiersza
    X_normalized = inv_norms.dot(X)

    return X_normalized


def find_similar_documents_abs_cosine(query_vector, X_normalized, k=5):
    """
    Znajduje k dokumentów najbardziej podobnych do zapytania używając
    wartości bezwzględnej miary kosinusowej.

    Miara podobieństwa: |cos θj| = |q^T * d_j| / (||q|| * ||d_j||)
    """
    # Normalizacja wektora zapytania
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return [], []  # Brak zapytania

    # Normalizacja wektora zapytania
    query_vector = query_vector / query_norm

    # Przygotowanie wektora zapytania do mnożenia macierzowego
    query_vector = query_vector.reshape(-1, 1)  # jako wektor kolumnowy

    # Obliczenie iloczynu skalarnego zapytania z każdym dokumentem: q^T * d_j
    # Macierz X_normalized już zawiera znormalizowane wektory dokumentów
    dot_products = X_normalized.dot(query_vector).flatten()

    # Obliczenie wartości bezwzględnej podobieństwa kosinusowego: |cos θj| = |q^T * d_j|
    # (Normy są równe 1 dzięki normalizacji, więc dzielenie nie jest potrzebne)
    abs_similarities = np.abs(dot_products)

    # Sortowanie dokumentów według podobieństwa (malejąco)
    most_similar_indices = np.argsort(abs_similarities)[::-1][:k]

    return most_similar_indices, abs_similarities[most_similar_indices]


def main():
    # Sprawdzenie dostępności plików
    try:
        # Wczytaj słownik
        print("Wczytywanie słownika...")
        with open("vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Wczytaj macierz TF-IDF
        print("Wczytywanie macierzy TF-IDF...")
        X_tfidf = sp.load_npz("tfidf_matrix.npz")

        # Wczytaj oryginalne dokumenty (aby wyświetlić wyniki)
        print("Wczytywanie oryginalnych dokumentów...")
        with open("wiki_documents.json", "r", encoding="utf-8") as f:
            original_docs = json.load(f)

        print(f"Normalizacja {X_tfidf.shape[0]} wektorów dokumentów...")
        # Normalizacja macierzy dokumentów
        X_normalized = normalize_matrix(X_tfidf)

        # Zapisz znormalizowaną macierz
        sp.save_npz("normalized_tfidf_matrix.npz", X_normalized)
        print("Zapisano znormalizowaną macierz do normalized_tfidf_matrix.npz")

        print(f"Gotowe do wyszukiwania w {len(original_docs)} dokumentach.")
    except FileNotFoundError as e:
        print(f"Błąd: Nie znaleziono potrzebnego pliku - {e}")
        print("Upewnij się, że uruchomiłeś najpierw skrypt compute_tfidf.py")
        return

    # Interfejs użytkownika dla wyszukiwania
    print("\n=== System wyszukiwania dokumentów (|cos θ|) ===")
    while True:
        query = input("\nWprowadź zapytanie (lub 'q' aby zakończyć): ")
        if query.lower() == 'q':
            break

        if not query.strip():
            print("Zapytanie jest puste. Spróbuj ponownie.")
            continue

        try:
            k = int(input("Ile dokumentów wyświetlić? [domyślnie 5]: ") or "5")
        except ValueError:
            k = 5

        # Przetwarzanie zapytania
        query_vector = process_query(query, vocab)

        # Sprawdzenie czy zapytanie zawiera znane słowa
        if np.sum(query_vector) == 0:
            print("Żadne słowo z zapytania nie występuje w słowniku. Spróbuj inne zapytanie.")
            continue

        # Wyszukiwanie podobnych dokumentów
        similar_indices, scores = find_similar_documents_abs_cosine(query_vector, X_normalized, k)

        # Wyświetlanie wyników
        if len(similar_indices) == 0:
            print("Nie znaleziono pasujących dokumentów.")
        else:
            print(f"\nZnaleziono {len(similar_indices)} dokumentów podobnych do zapytania '{query}':")
            for i, (idx, score) in enumerate(zip(similar_indices, scores), 1):
                # Przycięcie dokumentu dla czytelności
                doc_preview = original_docs[idx][:200] + "..." if len(original_docs[idx]) > 200 else original_docs[idx]
                print(f"\n{i}. Dokument #{idx} (podobieństwo |cos θ|: {score:.4f})")
                print(f"   {doc_preview}")


if __name__ == "__main__":
    main()