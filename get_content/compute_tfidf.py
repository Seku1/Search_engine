import json
import numpy as np
import scipy.sparse as sp


def compute_idf(X, vocab_size):
    """
    Oblicza wartości IDF dla każdego słowa w słowniku.

    IDF(w) = log(N/nw), gdzie:
    - N to całkowita liczba dokumentów
    - nw to liczba dokumentów zawierających słowo w
    """
    N = X.shape[0]  # liczba dokumentów
    df = np.bincount(X.nonzero()[1], minlength=vocab_size)  # liczba dokumentów dla każdego słowa
    # Unikamy dzielenia przez zero dodając 1 do licznika i mianownika
    idf = np.log(np.divide(N + 1, df + 1))
    return idf


def apply_idf_to_bow(X, idf):
    """
    Mnoży każdy element macierzy BoW przez odpowiednią wartość IDF
    """
    # Konwersja idf na macierz diagonalną
    idf_diag = sp.diags(idf, 0)
    # Mnożenie macierzy BoW przez IDF
    X_tfidf = X.dot(idf_diag)
    return X_tfidf


def main():
    # Wczytaj słownik
    print("Wczytywanie słownika...")
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Wczytaj macierz BoW
    print("Wczytywanie macierzy BoW...")
    X = sp.load_npz("bow_matrix.npz")

    # Oblicz wartości IDF
    print(f"Obliczanie IDF dla {len(vocab)} słów...")
    idf = compute_idf(X, len(vocab))

    # Zapisz wartości IDF do pliku
    np.save("idf_values.npy", idf)
    print("Zapisano wartości IDF do pliku idf_values.npy")

    # Zastosuj IDF do macierzy BoW
    print("Tworzenie macierzy TF-IDF...")
    X_tfidf = apply_idf_to_bow(X, idf)

    # Zapisz przekształconą macierz
    sp.save_npz("tfidf_matrix.npz", X_tfidf)
    print("Zapisano macierz TF-IDF do pliku tfidf_matrix.npz")

    print("\nPrzetwarzanie zakończone. Można teraz uruchomić search_documents.py aby wyszukiwać dokumenty.")


if __name__ == "__main__":
    main()