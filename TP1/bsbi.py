import os
import pickle
import contextlib
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, QueryParser, sort_diff_list, sort_intersect_list, sort_union_list, tokenize_text_by_regex, \
    remove_stop_words, stem_tokens
from compression import StandardPostings, VBEPostings, EliasGammaPostings
import contextlib
import heapq

""" 
Ingat untuk install tqdm terlebih dahulu
pip intall tqdm
"""
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_path(str): Path ke data
    output_path(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_path, output_path, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_path = data_path
        self.output_path = output_path
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_path, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_path, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_path, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_path, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def start_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_path in tqdm(sorted(next(os.walk(self.data_path))[1])):
            td_pairs = self.parsing_block(block_path)
            index_id = 'intermediate_index_'+block_path
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, path = self.output_path) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, path = self.output_path) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, path=self.output_path))
                               for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)

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
        termIDs dan docIDs. Dua variable ini harus persis untuk semua pemanggilan
        parse_block(...).
        """
        # TODO (FINISHED)

        td_pairs = []

        observed_block = os.path.join(self.data_path, block_path)

        for file_name in tqdm(sorted(next(os.walk(observed_block))[2])):
            file_path = os.path.join(observed_block, file_name)
            doc_id = self.doc_id_map[file_path]

            with open(file_path, 'r') as f:
                result = f.read()

                # Tokenization (using Regex, in TPK1)
                tokens = tokenize_text_by_regex(result)
                # Remove Stopwords (NLTK Stopwords)
                filtered_tokens = remove_stop_words(tokens, stopwords.words('english'))

                # Stemming (Lancaster Stemmer)
                stemmed_token = stem_tokens(filtered_tokens, LancasterStemmer())

                for term in stemmed_token:
                    term_id = self.term_id_map[term]
                    td_pair = (term_id, doc_id)

                    td_pairs.append(td_pair)
        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
            term_dict[term_id].add(doc_id)
        for term_id in sorted(term_dict.keys()):
            index.append(term_id, sorted(list(term_dict[term_id])))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

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
        # TODO (FINISHED)

        min_heap = []

        # External Merge Sort

        iterators = [index.__iter__() for index in indices]

        # In intermediate index, item stored is:
        # item[0] = dictionary of term_dict (termID, how many posting list, how many bytes it took)
        # item[1], list of terms
        for idx, iterator in enumerate(iterators):
            try:
                term_id, postings_list = next(iterator)
                heapq.heappush(min_heap, (term_id, idx, postings_list))
            except StopIteration:
                pass

        while min_heap:
            term_id, idx, postings_list = heapq.heappop(min_heap)

            while min_heap and min_heap[0][0] == term_id:
                _, j, other_postings = heapq.heappop(min_heap)
                postings_list = sorted(set(postings_list) | set(other_postings))

                try:
                    next_term_id, next_postings = next(iterators[j])
                    heapq.heappush(min_heap, (next_term_id, j, next_postings))
                except StopIteration:
                    pass

            # Write the merged postings to the final index
            merged_index.append(term_id, postings_list)

            try:
                next_term_id, next_postings = next(iterators[idx])
                heapq.heappush(min_heap, (next_term_id, idx, next_postings))
            except StopIteration:
                pass

    def boolean_retrieve(self, query):
        """
        Melakukan boolean retrieval untuk mengambil semua dokumen yang
        mengandung semua kata pada query. Lakukan pre-processing seperti
        yang telah dilakukan pada tahap indexing, kecuali *penghapusan stopwords*.

        Jika terdapat stopwords dalam query, return list kosong dan berikan pesan bahwa
        terdapat stopwords di dalam query.

        Parse query dengan class QueryParser. Ambil representasi postfix dari ekspresi
        untuk kemudian dievaluasi di method ini. Silakan baca pada URL di bawah untuk lebih lanjut.
        https://www.geeksforgeeks.org/evaluation-of-postfix-expression/

        Anda tidak wajib mengimplementasikan conjunctive queries optimization.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi. Ini dapat mengandung operator
            himpunan AND, NOT, dan DIFF, serta tanda kurung untuk presedensi. 

            contoh: (universitas AND indonesia OR depok) DIFF ilmu AND komputer

        Returns
        ------
        List[str]
            Daftar dokumen terurut yang mengandung sebuah query tokens.
            Harus mengembalikan EMPTY LIST [] jika tidak ada yang match.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        # TODO (ON PROGRESS)

        # Remove Stopwords (NLTK Stopwords)

        qp = QueryParser(query, LancasterStemmer(), stopwords.words('english'))

        if not qp.is_valid():
            print("Stopword was found in the query")
            return []

        query_postfix = qp.infix_to_postfix()

        self.load()

        with open(os.path.join(self.output_path, f"{self.index_name}.dict"), 'rb') as f:
            main_index = pickle.load(f)

        stack = []

        for token in query_postfix:
            if token in ['AND', 'OR', 'DIFF']:
                right = stack.pop()
                left = stack.pop()
                result = []
                if token == 'AND':
                    result = sort_intersect_list(left, right)
                elif token == 'OR':
                    result = sort_union_list(left, right)
                elif token == 'DIFF':
                    result = sort_diff_list(left, right)

                stack.append(result)
            else:
                term_id = self.term_id_map[token]
                try:
                    with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_path) as reader:
                        postings_list = reader.get_postings_list(term_id)
                except Exception:
                    postings_list = []

                stack.append(postings_list)

        if stack:
            result_doc_ids = stack.pop()
            return [self.doc_id_map[doc_id] for doc_id in result_doc_ids]
        else:
            return []


if __name__ == "__main__":
    BSBI_instance = BSBIIndex(data_path = 'arxiv_collections', \
                              postings_encoding = EliasGammaPostings, \
                              output_path = 'index_eg')
    BSBI_instance.start_indexing() # memulai indexing!

