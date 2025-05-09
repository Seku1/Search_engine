import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
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


def compute_svd_components(X, k):
    """
    Oblicza dekompozycję SVD ale zamiast tworzyć pełną aproksymację macierzy,
    zwraca tylko komponenty potrzebne do dalszych obliczeń.
    """
    print(f"Obliczanie SVD dla macierzy {X.shape} z k={k}...")

    # Używamy svds z scipy.sparse.linalg dla wydajnej obsługi dużych macierzy rzadkich
    U, s, Vt = svds(X, k=k)

    # svds sortuje wartości osobliwe od najmniejszej do największej,
    # więc musimy odwrócić kolejność
    U = U[:, ::-1]
    s = s[::-1]
    Vt = Vt[::-1, :]

    # Dla celów debugowania, wyświetl wartości osobliwe
    print("Największe wartości osobliwe:", s)

    return U, s, Vt


def find_similar_documents_svd(query_vector, U, s, Vt, k_docs=5):
    """
    Znajduje k dokumentów najbardziej podobnych do zapytania używając
    zdekomponowanej reprezentacji SVD bez explicite tworzenia macierzy A_k.

    Zamiast tego używa projekcji wektora zapytania na przestrzeń ukrytą,
    a następnie oblicza podobieństwa używając wartości osobliwych i macierzy U.
    """
    # Obliczanie normy wektora zapytania
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return [], []  # Brak zapytania

    # Normalizacja wektora zapytania
    query_vector = query_vector / query_norm

    # Projekcja zapytania na przestrzeń ukrytą
    # q_hat = query_vector @ Vt.T @ np.diag(s)
    # Implementacja wydajna pamięciowo:
    q_hat = np.zeros(len(s))
    for i in range(len(s)):
        q_hat[i] = np.dot(query_vector, Vt[i, :].T) * s[i]

    # Obliczanie podobieństwa do dokumentów w przestrzeni ukrytej
    # similarities = q_hat @ U.T
    # Implementacja wydajna pamięciowo:
    similarities = np.zeros(U.shape[0])
    for i in range(len(s)):
        similarities += q_hat[i] * U[:, i]

    # Obliczanie norm dokumentów w przestrzeni ukrytej
    # Zamiast obliczać normy z pełnej macierzy A_k, używamy wartości osobliwych i U
    doc_norms = np.zeros(U.shape[0])
    for i in range(len(s)):
        doc_norms += (s[i] * U[:, i]) ** 2
    doc_norms = np.sqrt(doc_norms)

    # Unikanie dzielenia przez zero
    doc_norms[doc_norms == 0] = 1.0

    # Normalizacja podobieństwa przez normy dokumentów
    similarities = similarities / doc_norms

    # Sortowanie dokumentów według podobieństwa (malejąco)
    most_similar_indices = np.argsort(similarities)[::-1][:k_docs]

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

        # Zastosuj SVD i redukcję rangi
        k_components = min(100, min(X_tfidf.shape) - 1)  # Domyślnie 100 lub mniej jeśli macierz jest mniejsza

        # Zapytaj użytkownika o liczbę komponentów SVD
        try:
            user_k = input(f"Podaj liczbę komponentów SVD [domyślnie {k_components}]: ")
            if user_k.strip():
                k_components = int(user_k)
        except ValueError:
            print(f"Niepoprawna wartość. Używam domyślnej wartości k={k_components}")

        # Oblicz komponenty SVD - już nie tworzymy pełnej macierzy A_k
        U, s, Vt = compute_svd_components(X_tfidf, k_components)

        print(f"Obliczono {k_components} głównych komponentów SVD")
        print(f"Gotowe do wyszukiwania w {len(original_docs)} dokumentach.")
    except FileNotFoundError as e:
        print(f"Błąd: Nie znaleziono potrzebnego pliku - {e}")
        print("Upewnij się, że uruchomiłeś najpierw skrypt compute_tfidf.py")
        return
    except ValueError as e:
        print(f"Błąd przy obliczaniu SVD: {e}")
        print("Spróbuj zmniejszyć liczbę komponentów SVD")
        return

    # Interfejs użytkownika dla wyszukiwania
    print("\n=== System wyszukiwania dokumentów (SVD, Low Rank Approximation) ===")
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

        # Wyszukiwanie podobnych dokumentów używając SVD bez tworzenia pełnej macierzy aproksymacji
        similar_indices, scores = find_similar_documents_svd(query_vector, U, s, Vt, k)

        # Wyświetlanie wyników
        if len(similar_indices) == 0:
            print("Nie znaleziono pasujących dokumentów.")
        else:
            print(f"\nZnaleziono {len(similar_indices)} dokumentów podobnych do zapytania '{query}':")
            for i, (idx, score) in enumerate(zip(similar_indices, scores), 1):
                # Przycięcie dokumentu dla czytelności
                doc_preview = original_docs[idx][:200] + "..." if len(original_docs[idx]) > 200 else original_docs[idx]
                print(f"\n{i}. Dokument #{idx} (podobieństwo cos φ: {score:.4f})")
                print(f"   {doc_preview}")


if __name__ == "__main__":
    main()