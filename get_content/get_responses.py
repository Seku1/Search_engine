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


def find_similar_documents(query_vector, X_tfidf, k=5):
    """
    Znajduje k dokumentów najbardziej podobnych do zapytania używając miary kosinusowej.
    Implementacja bez używania sklearn.

    Miara kosinusowa: cos θj = q^T * d_j / (||q|| * ||d_j||)
    """
    # Obliczanie normy wektora zapytania
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return [], []  # Brak zapytania

    # Normalizacja wektora zapytania
    query_vector = query_vector / query_norm

    # Przygotowanie wektora zapytania do mnożenia macierzowego
    query_vector = query_vector.reshape(-1, 1)  # jako wektor kolumnowy

    # Obliczenie iloczynu skalarnego zapytania z każdym dokumentem: q^T * d_j
    # X_tfidf to macierz, gdzie wiersze to dokumenty, a kolumny to słowa
    # Transponujemy X_tfidf, aby kolumny stały się dokumentami
    dot_products = X_tfidf.dot(query_vector).flatten()

    # Obliczenie norm dokumentów
    # Dla każdego dokumentu j: ||d_j|| = sqrt(sum(d_j[i]^2))
    # Można to zrobić efektywnie na całej macierzy
    doc_norms = np.sqrt((X_tfidf.power(2)).sum(axis=1).A1)

    # Unikanie dzielenia przez zero
    doc_norms[doc_norms == 0] = 1.0

    # Obliczenie podobieństwa kosinusowego: cos θj = q^T * d_j / (||q|| * ||d_j||)
    similarities = dot_products / doc_norms

    # Sortowanie dokumentów według podobieństwa (malejąco)
    most_similar_indices = np.argsort(similarities)[::-1][:k]

    return most_similar_indices, similarities[most_similar_indices]


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

        print(f"Gotowe do wyszukiwania w {len(original_docs)} dokumentach.")
    except FileNotFoundError as e:
        print(f"Błąd: Nie znaleziono potrzebnego pliku - {e}")
        print("Upewnij się, że uruchomiłeś najpierw skrypt compute_tfidf.py")
        return

    # Interfejs użytkownika dla wyszukiwania
    print("\n=== System wyszukiwania dokumentów ===")
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
        similar_indices, scores = find_similar_documents(query_vector, X_tfidf, k)

        # Wyświetlanie wyników
        if len(similar_indices) == 0:
            print("Nie znaleziono pasujących dokumentów.")
        else:
            print(f"\nZnaleziono {len(similar_indices)} dokumentów podobnych do zapytania '{query}':")
            for i, (idx, score) in enumerate(zip(similar_indices, scores), 1):
                # Przycięcie dokumentu dla czytelności
                doc_preview = original_docs[idx][:200] + "..." if len(original_docs[idx]) > 200 else original_docs[idx]
                print(f"\n{i}. Dokument #{idx} (podobieństwo: {score:.4f})")
                print(f"   {doc_preview}")


if __name__ == "__main__":
    main()