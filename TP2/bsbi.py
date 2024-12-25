import os
import pickle
import contextlib
import heapq
import math
from nltk.corpus import stopwords
from porter2stemmer import Porter2Stemmer

from index import InvertedIndexReader, InvertedIndexWriter
from trie import Trie
from util import IdMap, merge_and_sort_posts_and_tfs, tokenize_text_by_regex, stem_tokens, remove_stop_words
from compression import VBEPostings
from tqdm import tqdm


def check_if_pointers_are_not_on_the_end_of_postings(pointers, document_at_a_time_list):
    for idx in range(len(pointers)):
        if pointers[idx] < len(document_at_a_time_list[idx][0]):
            return True

    return False

def check_if_all_doc_ids_is_equal(docIDs):
    return all(docID == docIDs[0] for docID in docIDs if docID != float('inf'))

def preprocess_query(query: str):
    tokens = tokenize_text_by_regex(query)
    filtered_tokens = remove_stop_words(tokens, stopwords.words('english'))
    stemmed_token = stem_tokens(filtered_tokens, Porter2Stemmer())

    return stemmed_token

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    trie(Trie): Class Trie untuk query auto-completion
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.trie = Trie()

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map, term_id_map, dan trie ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        with open(os.path.join(self.output_dir, 'trie.pkl'), 'wb') as f:
            # file ini mungkin agak besar
            pickle.dump(self.trie, f)

    def load(self):
        """Memuat doc_id_map, term_id_map, dan trie dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'trie.pkl'), 'rb') as f:
            self.trie = pickle.load(f)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Anda bisa menggunakan stemmer bahasa Inggris yang tersedia, seperti Porter Stemmer
        https://github.com/evandempsey/porter2-stemmer

        Untuk membuang stopwords, Anda dapat menggunakan library seperti NLTK.

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO (FINISHED)
        # Hint: Anda dapat mengisi trie di sini

        td_pairs = []

        observed_block = os.path.join(self.data_dir, block_path)

        for file_name in tqdm(sorted(next(os.walk(observed_block))[2])):
            file_path = os.path.join(observed_block, file_name)
            doc_id = self.doc_id_map[file_path]

            with open(file_path, 'r') as f:
                result = f.read()

                # Tokenization (using Regex, in TPK1)
                tokens = tokenize_text_by_regex(result)
                # Remove Stopwords (NLTK Stopwords)
                filtered_tokens = remove_stop_words(tokens, stopwords.words('english'))

                # Implement trie here
                for token in filtered_tokens:
                    self.trie.insert(token, 1)

                # Stemming (Porter2 Stemmer)
                stemmed_token = stem_tokens(filtered_tokens, Porter2Stemmer())

                for term in stemmed_token:
                    term_id = self.term_id_map[term]
                    td_pair = (term_id, doc_id)

                    td_pairs.append(td_pair)

        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # term_dict merupakan dictionary yang berisi dictionary yang
        # melakukan mapping dari doc_id ke tf
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = dict()
            # Mengupdate juga TF (yang merupakan value dari dictionary yang di dalam)
            term_dict[term_id][doc_id] = term_dict[term_id].get(doc_id, 0) + 1
        
        for term_id in sorted(term_dict.keys()):
            # Sort postings list (dan tf list yang bersesuaian)
            sorted_postings_tf = dict(sorted(term_dict[term_id].items()))
            # Postings list adalah keys, TF list adalah values
            index.append(term_id, list(sorted_postings_tf.keys()), 
                         list(sorted_postings_tf.values()))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def _compute_score_tfidf(self, tf, df, N):
        """
        Fungsi ini melakukan komputasi skor TF-IDF.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        score = w(t, Q) x w(t, D)
        Tidak perlu lakukan normalisasi pada score.

        Gunakan log basis 10.

        Parameters
        ----------
        tf: int
            Term frequency.

        df: int
            Document frequency.

        N: int
            Jumlah dokumen di corpus. 

        Returns
        -------
        float
            Skor hasil perhitungan TF-IDF.
        """
        # TODO (FINISHED)

        w_document = 0

        if tf > 0:
            w_document = 1 + math.log10(tf)

        w_query = math.log10(N/df)

        score = w_query * w_document

        return score
    
    def _compute_score_bm25(self, tf, df, N, k1, b, dl, avdl):
        """
        Fungsi ini melakukan komputasi skor BM25.
        Gunakan log basis 10 dan tidak perlu lakukan normalisasi.
        Silakan lihat penjelasan parameters di slide.

        Returns
        -------
        float
            Skor hasil perhitungan TF-IDF.
        """
        # TODO (FINISHED)
        return math.log10(N/df) * ( (k1 + 1) * tf / (k1 * ((1-b) + (b*dl)/avdl) + tf) )

    def retrieve_tfidf_daat(self, query, k=10):
        """
        Lakukan retrieval TF-IDF dengan skema DaaT.
        Method akan mengembalikan top-K retrieval results.

        Program tidak perlu paralel sepenuhnya. Untuk mengecek dan mengevaluasi
        dokumen yang di-point oleh pointer pada waktu tertentu dapat dilakukan
        secara sekuensial, i.e., seperti menggunakan for loop.

        Beberapa informasi penting:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """

        # TODO (FINISHED)
        query_list = preprocess_query(query)

        scores = dict()

        # For now will make a list that consists of:
        # postings_list, tf_list, df
        document_at_a_time_list = []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as reader:
            N = len(reader.doc_length)

            for term in query_list:
                if term in self.term_id_map:
                    term_id = self.term_id_map[term]
                    postings_list, tf_list = reader.get_postings_list(term_id)
                    df = reader.postings_dict[term_id][1]
                    document_at_a_time_list.append((postings_list, tf_list, df))

            pointers = [0]*len(document_at_a_time_list)

            while check_if_pointers_are_not_on_the_end_of_postings(pointers, document_at_a_time_list):
                # 0 here stands for postings_list
                docIDs = [document_at_a_time_list[i][0][pointers[i]] if pointers[i] < len(
                    document_at_a_time_list[i][0]) else float('inf') for i in range(len(document_at_a_time_list))]

                if check_if_all_doc_ids_is_equal(docIDs):
                    temp_result = 0

                    for pointerIdx in range(len(pointers)):
                        specifiedIdx = pointers[pointerIdx]

                        if specifiedIdx < len(document_at_a_time_list[pointerIdx][0]):
                            tf_list = document_at_a_time_list[pointerIdx][1]
                            df = document_at_a_time_list[pointerIdx][2]
                            tf = tf_list[specifiedIdx]

                            temp_result += self._compute_score_tfidf(tf, df, N)
                            pointers[pointerIdx] += 1

                    scores[docIDs[0]] = temp_result
                else:
                    min_doc_id = min(docID for docID in docIDs if docID != float('inf'))

                    for i in range(len(document_at_a_time_list)):
                        specifiedIdx = pointers[i]
                        if docIDs[i] == min_doc_id and specifiedIdx < len(document_at_a_time_list[i][0]):
                            postings_list=document_at_a_time_list[i][0]
                            tf_list=document_at_a_time_list[i][1]
                            df=document_at_a_time_list[i][2]
                            tf = tf_list[specifiedIdx]

                            if specifiedIdx < len(postings_list):
                                temp_result = self._compute_score_tfidf(tf, df, N)
                                pointers[i] += 1

                                if docIDs[i] in scores:
                                    scores[docIDs[i]] += temp_result
                                else:
                                    scores[docIDs[i]] = temp_result

        return self.get_top_k_by_score(scores, k)

    def retrieve_tfidf_taat(self, query, k=10):
        """
        Lakukan retrieval TF-IDF dengan skema TaaT.
        Method akan mengembalikan top-K retrieval results.

        Beberapa informasi penting: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO (FINISHED)
        query_list = preprocess_query(query)
        scores = dict()

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as reader:
            N = len(reader.doc_length)

            for term in query_list:
                if term in self.term_id_map:
                    term_id = self.term_id_map[term]
                    postings_list, tf_list = reader.get_postings_list(term_id)
                    df = reader.postings_dict[term_id][1]

                    for idx in range(len(postings_list)):
                        tf = tf_list[idx]
                        score_document = self._compute_score_tfidf(tf, df, N)

                        docID = postings_list[idx]

                        if docID in scores:
                            scores[docID] += score_document
                        else:
                            scores[docID] = score_document

        return self.get_top_k_by_score(scores, k)

    def retrieve_bm25_taat(self, query, k=10, k1=1.2, b=0.75):
        """
        Lakukan retrieval BM-25 dengan skema TaaT.
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        # TODO (FINSIHED)

        query_list = preprocess_query(query)
        scores = dict()

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as reader:
            # Get avdl (Average Document Length)
            N = len(reader.doc_length)
            avdl = reader.get_average_document_length()

            for term in query_list:
                if term in self.term_id_map:
                    term_id = self.term_id_map[term]
                    postings_list, tf_list = reader.get_postings_list(term_id)
                    df = reader.postings_dict[term_id][1]

                    for idx in range(len(postings_list)):
                        tf = tf_list[idx]
                        docID = postings_list[idx]

                        # Get dl (Document Length)
                        dl = reader.doc_length[docID]

                        score_document = self._compute_score_bm25(tf, df, N, k1, b, dl, avdl)

                        if docID in scores:
                            scores[docID] += score_document
                        else:
                            scores[docID] = score_document

        return self.get_top_k_by_score(scores, k)

    def retrieve_tfidf_wand(self, query, k = 10):
        """
        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """

        query_terms = preprocess_query(query)
        term_postings = {}
        upper_bounds = {}

        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as reader:
            N = len(reader.doc_length)

            for term in query_terms:
                try:
                    term_id = self.term_id_map[term]
                    postings_list, tf_list = reader.get_postings_list(term_id)
                    df = reader.postings_dict[term_id][1]

                    # Maximum term frequency
                    max_tf = max(tf_list)
                    term_postings[term] = list(zip(postings_list, tf_list))
                    upper_bounds[term] = self._compute_score_tfidf(max_tf, df, N)

                except KeyError:
                    continue

        candidates = []
        pointers = {term: 0 for term in query_terms}
        current_threshold = 0

        while True:
            candidate_doc = float("inf")
            for term, postings in term_postings.items():
                if pointers[term] < len(postings):
                    candidate_doc = min(candidate_doc, postings[pointers[term]][0])

            if candidate_doc == float("inf"):
                break

            full_score = 0
            for term, postings in term_postings.items():
                if pointers[term] < len(postings) and postings[pointers[term]][0] == candidate_doc:
                    term_id = self.term_id_map[term]
                    df = reader.postings_dict[term_id][1]

                    _, tf = postings[pointers[term]]

                    full_score += self._compute_score_tfidf(tf, df, N)
                    pointers[term] += 1

            if full_score > current_threshold:
                heapq.heappush(candidates, (full_score, candidate_doc))
                if len(candidates) > k:
                    heapq.heappop(candidates)
                    current_threshold = candidates[0][0]

        top_k_docs = []

        while candidates:
            score, doc_id = heapq.heappop(candidates)
            doc_str = self.doc_id_map.id_to_str[doc_id]
            top_k_docs.append((score, doc_str))

        top_k_docs.reverse()

        return top_k_docs

    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)

    def get_top_k_by_score(self, score_docs, k):
        """
        Method ini berfungsi untuk melakukan sorting terhadap dokumen berdasarkan score
        yang dihitung, lalu mengembalikan top-k dokumen tersebut dalam bentuk tuple
        (score, document). Silakan gunakan heap agar lebih efisien.

        Parameters
        ----------
        score_docs: Dictionary[int -> float]
            Dictionary yang berisi mapping docID ke score masing-masing dokumen tersebut.

        k: Int
            Jumlah dokumen yang ingin di-retrieve berdasarkan score-nya.

        Result
        -------
        List[(float, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        # TODO (FINISHED)

        min_heap = []

        for doc_id, score in score_docs.items():
            if len(min_heap) < k:
                heapq.heappush(min_heap, (score, doc_id))

            elif score > min_heap[0][0]:
                heapq.heapreplace(min_heap, (score, doc_id))

        top_k_docs = []

        while min_heap:
            score, doc_id = heapq.heappop(min_heap)
            doc_str = self.doc_id_map.id_to_str[doc_id]
            top_k_docs.append((score, doc_str))

        top_k_docs.reverse()

        return top_k_docs
    
    def get_query_recommendations(self, query, k=5):
        # Method untuk mendapatkan rekomendasi untuk QAC
        # Tidak perlu mengubah ini
        self.load()
        last_token = query.split()[-1]
        recc = self.trie.get_recommendations(last_token, k)
        return recc

if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='arxiv_collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!