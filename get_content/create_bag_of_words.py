import json
import numpy as np
from collections import Counter
import scipy.sparse as sp

# 1. Wczytaj oczyszczone dokumenty
with open("wiki_procesed.json", "r", encoding="utf-8") as f:
    processed_docs = json.load(f)

# 2. Zbuduj słownik: token -> indeks
vocab = {}
for doc in processed_docs:
    for tok in doc.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)

print(f"Zbudowano słownik o rozmiarze: {len(vocab)} tokenów.")

# 3a. Wariant: lista Counterów (lekka, nie zużywa pamięci na pełną macierz)
doc_counters = [Counter(doc.split()) for doc in processed_docs]

# Teraz doc_counters[i][tok] to liczba wystąpień tok w dokumencie i

# 3b. Wariant: rzadka macierz SciPy CSR (D x M)
D = len(processed_docs)
M = len(vocab)

rows = []
cols = []
data = []

for i, counter in enumerate(doc_counters):
    for tok, cnt in counter.items():
        j = vocab[tok]
        rows.append(i)
        cols.append(j)
        data.append(cnt)

# zbuduj macierz D×M
X = sp.csr_matrix((data, (rows, cols)), shape=(D, M), dtype=int)

print(f"Macierz bag-of-words ma kształt: {X.shape} i zawiera {X.nnz} niezerowych elementów.")

# 4. Zapisz słownik i opcjonalnie macierz
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

# Jeśli chcesz zapisać macierz do pliku .npz:
sp.save_npz("bow_matrix.npz", X)

print("Zapisano: vocab.json oraz bow_matrix.npz")
