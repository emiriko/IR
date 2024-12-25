# BM-25 dengan konfigurasi parameter tertentu bisa diubah behavior-nya
# menjadi seperti TF-IDF
from itertools import product

# Hyperparameter dari BM-25 yang bisa diubah-ubah adalah k1 dan b
# Silakan menentukan opsi k1 dan b Anda sendiri

from bsbi import BSBIIndex
from compression import VBEPostings

BSBI_instance = BSBIIndex(data_dir='arxiv_collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

BSBI_instance.load()
# Isi dengan kandidat hyperparameter yang Anda inginkan
k1_candidates = [0.2, 0.5, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
b_candidates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

query = "neural network"

# TODO: Lakukan hyperparameter tuning (FINISHED)

def get_matching_docs(tf_idf, bm25):
    """
        Computes the number of exact matches and loose matches between two lists of documents.

        Parameters:
        - tf_idf_docs: List of documents from TF-IDF ranked results
        - bm25_docs: List of documents from BM25 ranked results

        Returns:
        - exact_matches: Number of documents that match in the same position
        - loose_matches: Number of documents that match but in different positions
    """

    exact_matches = 0
    loose_matches = 0

    for i, doc in enumerate(bm25):
        if i < len(tf_idf) and tf_idf[i] == doc:
            exact_matches += 1
        elif doc in tf_idf:
            loose_matches += 1

    return exact_matches, loose_matches

def tune_bm25_hyperparameters(target_value):
    tf_idf_docs = [docStr for score, docStr in target_value]
    for k1, b in product(k1_candidates, b_candidates):
        bm25_result = BSBI_instance.retrieve_bm25_taat(query, 20, k1, b)
        bm25_docs = [docStr for score, docStr in bm25_result]

        exact_matches, loose_matches = get_matching_docs(tf_idf_docs, bm25_docs)

        print(f"BM25 with k1={k1}, b={b} -> Exact matches: {exact_matches}, Loose matches: {loose_matches}")

target = BSBI_instance.retrieve_tfidf_taat(query, 20)
tune_bm25_hyperparameters(target)