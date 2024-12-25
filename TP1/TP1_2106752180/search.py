from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_path = 'arxiv_collections', \
                          postings_encoding = VBEPostings, \
                          output_path = 'index_vb')

queries = ["(cosmological AND (quantum OR continuum)) AND geodesics"]
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for doc in BSBI_instance.boolean_retrieve(query):
        print(doc)
    print()