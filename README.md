# Sprawozdanie z Projektu: System Wyszukiwania Dokumentów Tekstowych

## 1. Wprowadzenie i Cel Projektu

Celem projektu było zaimplementowanie systemu wyszukiwania dokumentów tekstowych, który umożliwia odnajdywanie dokumentów najbardziej relevantnych dla zadanego zapytania użytkownika. System opiera się na modelu przestrzeni wektorowej (Vector Space Model), wykorzystując macierze TF-IDF (Term Frequency-Inverse Document Frequency) do reprezentacji dokumentów i zapytań. Dodatkowo, zaimplementowano możliwość wyszukiwania w zredukowanej przestrzeni cech uzyskanej za pomocą dekompozycji SVD (Singular Value Decomposition), znaną jako LSA (Latent Semantic Analysis). Aplikacja webowa została stworzona przy użyciu frameworka Flask, aby zapewnić interaktywny interfejs użytkownika.

Projekt obejmuje następujące etapy:
1.  Pobieranie i wstępne przetwarzanie dokumentów (artykułów z Wikipedii).
2.  Tworzenie słownika i macierzy BoW (Bag-of-Words).
3.  Obliczanie macierzy TF-IDF.
4.  Implementacja różnych metod wyszukiwania:
    *   Standardowe podobieństwo kosinusowe.
    *   Podobieństwo oparte na wartości bezwzględnej kosinusa (dla znormalizowanych wektorów).
    *   Wyszukiwanie w przestrzeni LSA (SVD).
5.  Stworzenie interfejsu webowego do interakcji z systemem.

## 2. Opis Komponentów Systemu (Skrypty Python)

System składa się z kilku skryptów Pythona, każdy odpowiedzialny za określoną część przetwarzania danych lub funkcjonalności.

### 2.1. `get_wiki.py`
*   **Cel:** Pobieranie treści artykułów z Wikipedii na podstawie zadanej kategorii.
*   **Działanie:** Wykorzystuje API MediaWiki do pobrania listy tytułów artykułów z danej kategorii, a następnie pobiera pełną treść tych artykułów. Tekst jest wstępnie czyszczony (usuwanie nagłówków HTML, nadmiarowych nowych linii). Wynik zapisywany jest do pliku `wiki_documents.json`.
*   **Parametry:** Możliwość zdefiniowania kategorii, limitu artykułów, liczby wątków roboczych oraz nazwy pliku wyjściowego poprzez argumenty linii poleceń.

### 2.2. `preproces.py`
*   **Cel:** Wstępne przetwarzanie tekstu (preprocessing).
*   **Działanie:** Wczytuje surowe dokumenty z `wiki_documents.json`. Dla każdego dokumentu wykonuje:
    1.  Konwersję do małych liter.
    2.  Usunięcie znaków niealfanumerycznych (zachowując spacje).
    3.  Tokenizację.
    4.  Usunięcie słów stopu (stopwords) i słów krótszych niż 3 znaki.
    5.  Stemming (sprowadzanie słów do ich rdzenia) za pomocą SnowballStemmer.
*   Przetworzone dokumenty są zapisywane do pliku `wiki_procesed.json`. Wymaga pobrania zasobów NLTK (stopwords).

### 2.3. `create_bag_of_words.py`
*   **Cel:** Stworzenie słownika oraz macierzy Bag-of-Words (BoW).
*   **Działanie:** Wczytuje przetworzone dokumenty z `wiki_procesed.json`.
    1.  Tworzy słownik (mapowanie słowo -> indeks) na podstawie wszystkich słów występujących w korpusie, filtrując te, które pojawiają się mniej niż 5 razy. Słownik zapisywany jest do `vocab.json`.
    2.  Tworzy rzadką macierz BoW (liczba dokumentów x rozmiar słownika), gdzie każda komórka (i, j) zawiera liczbę wystąpień słowa j w dokumencie i. Macierz zapisywana jest jako `bow_matrix.npz`.

### 2.4. `compute_tfidf.py`
*   **Cel:** Obliczenie macierzy TF-IDF.
*   **Działanie:** Wczytuje słownik (`vocab.json`) oraz macierz BoW (`bow_matrix.npz`).
    1.  Oblicza wartości IDF (Inverse Document Frequency) dla każdego słowa w słowniku: `idf(w) = log((N+1)/(nw+1))`, gdzie N to liczba dokumentów, a nw to liczba dokumentów zawierających słowo w. Wartości IDF zapisywane są do `idf_values.npy`.
    2.  Przekształca macierz BoW w macierz TF-IDF poprzez przemnożenie częstości terminów (TF, zawartych w BoW) przez odpowiednie wartości IDF. Wynikowa macierz TF-IDF jest zapisywana jako `tfidf_matrix.npz`.

### 2.5. `normalized_search.py` (CLI i generator macierzy)
*   **Cel:** Implementacja wyszukiwania opartego na wartości bezwzględnej podobieństwa kosinusowego oraz generowanie znormalizowanej macierzy TF-IDF.
*   **Działanie:**
    1.  Wczytuje macierz TF-IDF.
    2.  Normalizuje każdy wiersz (wektor dokumentu) macierzy TF-IDF, tak aby miał długość 1. Znormalizowana macierz jest zapisywana jako `normalized_tfidf_matrix.npz`.
    3.  Umożliwia wyszukiwanie dokumentów z linii poleceń, gdzie podobieństwo jest liczone jako `|q^T * d_j|` (ponieważ `||q||` i `||d_j||` są równe 1 po normalizacji).

### 2.6. `get_responses.py` (CLI)
*   **Cel:** Implementacja wyszukiwania opartego na standardowym podobieństwie kosinusowym.
*   **Działanie:** Wczytuje macierz TF-IDF i słownik. Umożliwia użytkownikowi wprowadzanie zapytań z linii poleceń. Dla każdego zapytania:
    1.  Przetwarza zapytanie na wektor TF (normalizowany).
    2.  Oblicza podobieństwo kosinusowe między wektorem zapytania a wszystkimi wektorami dokumentów w macierzy TF-IDF.
    3.  Wyświetla k najbardziej podobnych dokumentów.

### 2.7. `svd_search.py` (CLI i generator macierzy SVD)
*   **Cel:** Implementacja wyszukiwania w przestrzeni LSA (SVD) oraz generowanie macierzy SVD.
*   **Działanie:**
    1.  Wczytuje macierz TF-IDF.
    2.  Pozwala użytkownikowi zdefiniować liczbę wymiarów `k` dla dekompozycji SVD.
    3.  Oblicza macierze `Uk`, `Sk` (jako wektor wartości osobliwych), `Vtk` z `svds(X_tfidf, k=k_dimensions)`.
    4.  Zapisuje macierze SVD do podkatalogu `svd_matrices/` (np. `svd_Uk_100.npy`).
    5.  Umożliwia wyszukiwanie dokumentów z linii poleceń. Zapytanie jest rzutowane na przestrzeń SVD (`q_svd = q_norm * Vtk.T / Sk`), a następnie obliczane jest podobieństwo kosinusowe z dokumentami w tej przestrzeni (`doc_vectors_svd = Uk * Sk`).

### 2.8. `app.py` (Aplikacja Webowa Flask)
*   **Cel:** Udostępnienie interfejsu webowego do wyszukiwania dokumentów i zarządzania SVD.
*   **Działanie:** Aplikacja Flask, która przy starcie ładuje wszystkie niezbędne dane (`vocab.json`, `tfidf_matrix.npz`, `wiki_documents.json`, opcjonalnie `normalized_tfidf_matrix.npz` i domyślne macierze SVD).
*   **Funkcjonalności:**
    *   Strona główna (`/`) z polem do wprowadzania zapytania, wyborem liczby wyników oraz typu wyszukiwania (standardowe, z wartością bezwzględną kosinusa, SVD).
    *   Endpoint `/search` (POST) obsługujący zapytania, przetwarzający je i zwracający wyniki w formacie JSON.
    *   Endpoint `/document/<doc_id>` (GET) zwracający pełną treść dokumentu o danym ID.
    *   Endpoint `/svd-dimensions` (GET) listujący dostępne wymiarowości SVD i aktualnie załadowaną.
    *   Endpoint `/compute-svd` (POST) pozwalający na obliczenie i zapisanie macierzy SVD dla zadanej liczby wymiarów.
    *   Automatyczne generowanie `normalized_tfidf_matrix.npz` przy pierwszym ładowaniu, jeśli plik nie istnieje.
    *   Dynamiczne ładowanie macierzy SVD o różnych wymiarowościach na żądanie użytkownika.

### 2.9. `cos.py`
*   **Cel:** Alternatywny skrypt do przetwarzania wstępnego.
*   **Działanie:** Zawiera funkcję `preprocess` podobną do tej w `preproces.py`. Wydaje się być wcześniejszą wersją lub eksperymentalnym skryptem, ponieważ główny przepływ pracy opiera się na `preproces.py`. Zapisuje wynik do `wiki_procesed.json`.

## 3. Struktura Danych (Generowane Pliki)

Podczas działania systemu generowane i wykorzystywane są następujące pliki:

*   `wiki_documents.json`: Lista surowych tekstów artykułów pobranych z Wikipedii.
*   `wiki_procesed.json`: Lista dokumentów po preprocessingu (stemming, stopwords, etc.).
*   `vocab.json`: Słownik mapujący słowa na unikalne indeksy liczbowe.
*   `bow_matrix.npz`: Rzadka macierz Bag-of-Words.
*   `idf_values.npy`: Wektor wartości IDF dla każdego słowa w słowniku.
*   `tfidf_matrix.npz`: Rzadka macierz TF-IDF.
*   `normalized_tfidf_matrix.npz`: Macierz TF-IDF z wierszami znormalizowanymi do długości 1.
*   `svd_matrices/`: Katalog zawierający macierze SVD:
    *   `svd_Uk_DIM.npy`: Macierz lewych wektorów osobliwych (dokumenty w przestrzeni SVD).
    *   `svd_Sk_DIM.npy`: Wektor wartości osobliwych.
    *   `svd_Vtk_DIM.npy`: Macierz prawych wektorów osobliwych (słowa w przestrzeni SVD).
    *   `svd_config.json`: Przechowuje domyślną/ostatnio używaną liczbę wymiarów SVD.

## 4. Instrukcja Uruchomienia

### 4.1. Wymagania
*   Python 3.x
*   Biblioteki (zainstaluj używając `pip install -r requirements.txt`):
    *   `flask`
    *   `numpy`
    *   `scipy`
    *   `requests`
    *   `nltk`
    *   `tqdm`

    Plik `requirements.txt` powinien zawierać:
    ```
    flask
    numpy
    scipy
    requests
    nltk
    tqdm
    ```

### 4.2. Pobranie zasobów NLTK
Uruchom interpreter Pythona i wykonaj:
```python
import nltk
nltk.download('stopwords')