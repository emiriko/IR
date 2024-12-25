from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='arxiv_collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

query = input("Masukkan query Anda: ")

# TODO (FINISHED)
# silakan dilanjutkan sesuai contoh interaksi program pada dokumen soal

print("Rekomendasi query yang sesuai:")

recommendations = BSBI_instance.get_query_recommendations(query)

print(f"1. {query}")
for idx in range(len(recommendations)):
    end_recommendation = query + recommendations[idx]
    print(f"{idx+2}. {end_recommendation}")

choosen_option = int(input("Masukkan nomor query yang Anda maksud: "))
idx_query = choosen_option - 2

print("")

actual_query = query

if idx_query != -1:
    actual_query += recommendations[idx_query]

print(f"Pilihan Anda adalah '{actual_query}'.")
result = BSBI_instance.retrieve_tfidf_taat(actual_query)
print("Hasil pencarian:")

for score, docStr in result:
    print(f"{docStr} {score}")


