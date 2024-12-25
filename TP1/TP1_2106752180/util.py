import re
from abc import ABC, abstractmethod
from typing import Optional, List
from nltk.stem import LancasterStemmer

class StemInterface(ABC):
    @abstractmethod
    def stem(self, s: str):
        pass

class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        # TODO (FINISHED)
        return len(self.id_to_str)

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        # TODO (FINISHED)

        if s in self.str_to_id:
            return self.str_to_id[s]
        else:
            self.str_to_id[s] = len(self.id_to_str)
            self.id_to_str.append(s)
            return len(self.id_to_str) - 1
    
    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        # TODO (FINISHED)
        return self.id_to_str[i]

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        """
        # TODO (FINISHED)
        if type(key) == str:
            return self.__get_id(key)
        else:
            return self.__get_str(key)

class QueryParser:
    """
    Class untuk melakukan parsing query untuk boolean search
    
    Parameters
    ----------
    query: str
        Query string yang akan di-parse. Input dijamin valid, tidak ada imbalanced parentheses.
        Tanda kurung buka dijamin "menempel" di awal kata yang mengikuti (atau tanda kurung buka lainnya) dan
        tanda kurung tutup dijamin "menempel" di akhir kata yang diikuti (atau tanda kurung tutup lainnya).
        Sebagai contoh, bisa lihat pada contoh method query_string_to_list() atau pada test case.
    stemmer
        Objek stemmer untuk stemming token
    stopwords: set
        Set yang berisi stopwords
    """
    def __init__(self, query: str, stemmer: StemInterface, stopwords: set):
        self.special_token = ["AND", "OR", "DIFF", "(", ")"]
        self.query = query
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.token_list = self.__query_string_to_list()
        self.token_preprocessed = self.__preprocess_tokens()

    
    def is_valid(self):
        """
        Gunakan method ini untuk validasi query saat melakukan boolean retrieval,
        untuk menentukan apakah suatu query valid (tidak mengandung stopwords) atau tidak.
        """
        for token in self.token_list:
            if token in self.stopwords:
                return False
        return True

    def __query_string_to_list(self):
        """
        Melakukan parsing query dari yang berbentuk string menjadi list of tokens.
        Contoh: "term1 AND term2 OR (term3 DIFF term4)" --> ["term1", "AND", "term2", "OR", "(",
                                                             "term3", "DIFF", "term4", ")"]

        Returns
        -------
        List[str]
            query yang sudah di-parse
        """   
        # TODO (FINISHED)
        regex_pattern = """\(|\)|AND|OR|DIFF|\w+"""

        return re.findall(regex_pattern, self.query)

    def __preprocess_tokens(self):
        """
        Melakukan pre-processing pada query input, cukup lakukan stemming saja.
        Asumsikan bahwa tidak ada stopwords yang diberikan pada query input.
        Jangan lakukan pre-processing pada token spesial ('AND', 'OR', 'DIFF', '(', ')')
        
        Returns
        -------
        List[str]
            Daftar token yang telah di-preprocess
        """
        # TODO (FINISHED)

        result = []

        tokens = self.token_list
        for token in tokens:
            if token not in self.special_token:
                result.append(self.stemmer.stem(token))
            else:
                result.append(token)

        return result

    def infix_to_postfix(self):
        """
        Fungsi ini mengubah ekspresi infix menjadi postfix dengan menggunakan Algoritma Shunting-Yard. 
        Evaluasi akan dilakukan secara postfix juga. Gunakan tokens yang sudah di-pre-processed.
        Contoh: "A AND B" (infix) --> ["A", "B", "AND"] (postfix)
        Untuk selengkapnya, silakan lihat algoritma berikut: 
        https://www.geeksforgeeks.org/convert-infix-expression-to-postfix-expression/

        Returns
        -------
        list[str]
            list yang berisi token dalam ekspresi postfix
        """
        # TODO (FINISHED)

        infix = self.token_preprocessed

        postfix = []
        stack = []

        for token in infix:
            if token not in self.special_token:
                postfix.append(token)
            elif token == "(":
                stack.append(token)
            elif token == ")":
                while len(stack) != 0 and stack[-1] != "(":
                    postfix.append(stack.pop())
                if len(stack) > 0:
                    stack.pop()
            else:
                if len(stack) > 0:
                    while len(stack) != 0 and stack[-1] != "(":
                        postfix.append(stack.pop())

                    if stack[-1] == "(":
                        stack.append(token)
                        continue
                    postfix.append(token)
                else:
                    stack.append(token)

        while len(stack) != 0:
            postfix.append(stack.pop())

        return postfix

def sort_intersect_list(list_A, list_B):
    """
    Intersects two (ascending) sorted lists and returns the sorted result
    Melakukan Intersection dua (ascending) sorted lists dan mengembalikan hasilnya
    yang juga terurut.

    Parameters
    ----------
    list_A: List[Comparable]
    list_B: List[Comparable]
        Dua buah sorted list yang akan di-intersect.

    Returns
    -------
    List[Comparable]
        intersection yang sudah terurut
    """

    # TODO (FINISHED)

    set_A = set(list_A)
    set_B = set(list_B)

    return sorted(list(set_A.intersection(set_B)))

def sort_union_list(list_A, list_B):
    """
    Melakukan union dua (ascending) sorted lists dan mengembalikan hasilnya
    yang juga terurut.

    Parameters
    ----------
    list_A: List[Comparable]
    list_B: List[Comparable]
        Dua buah sorted list yang akan di-union.

    Returns
    -------
    List[Comparable]
        union yang sudah terurut
    """

    # TODO (FINISHED)

    set_A = set(list_A)
    set_B = set(list_B)

    return sorted(list(set_A.union(set_B)))

def sort_diff_list(list_A, list_B):
    """
    Melakukan difference dua (ascending) sorted lists dan mengembalikan hasilnya
    yang juga terurut.

    Parameters
    ----------
    list_A: List[Comparable]
    list_B: List[Comparable]
        Dua buah sorted list yang akan di-difference.

    Returns
    -------
    List[Comparable]
        difference yang sudah terurut
    """
    # TODO (FINISHED)

    set_A = set(list_A)
    set_B = set(list_B)

    return sorted(list(set_A.difference(set_B)))

def tokenize_text_by_regex(text: str, pattern: Optional[str] = r'\b\w+\b'):
    """
    Tokenisasi teks berdasarkan pola tokenizer_pattern
    """

    tokens = re.findall(pattern, text)

    return tokens

def remove_stop_words(tokens: List[str], stop_words: set[str]):
    tokens_without_stop_words = [token for token in tokens if token not in stop_words]
    return tokens_without_stop_words

def stem_tokens(tokens: List[str], stemmer: any):
    """
    Melakukan stemming pada tokens
    """
    stemmed_tokens: List[str] = [
        stemmer.stem(token) if token else "" for token in tokens
    ]

    stemmed_tokens_without_empty_string: List[str] = [
        token for token in stemmed_tokens if not ((token == "") or (token is None))
    ]
    return stemmed_tokens_without_empty_string

if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "docs_id salah"
    
    assert sort_intersect_list([1, 2, 3], [2, 3]) == [2, 3], "sorted_intersect salah"
    assert sort_intersect_list([4, 5], [1, 4, 7]) == [4], "sorted_intersect salah"
    assert sort_intersect_list([], []) == [], "sorted_intersect salah"

    assert sort_union_list([1, 2, 3], [2, 3]) == [1, 2, 3], "sorted_union salah"
    assert sort_union_list([4, 5], [1, 4, 7]) == [1, 4, 5, 7], "sorted_union salah"
    assert sort_union_list([], []) == [], "sorted_union salah"

    assert sort_diff_list([1, 2, 3], [2, 3]) == [1], "sorted_diff salah"
    assert sort_diff_list([4, 5], [1, 4, 7]) == [5], "sorted_diff salah"
    assert sort_diff_list([], []) == [], "sorted_diff salah"

    from porter2stemmer import Porter2Stemmer
    qp = QueryParser("((term1 AND term2) OR term3) DIFF (term6 AND (term4 OR term5) DIFF (term7 OR term8))", 
                     Porter2Stemmer(), set())
    assert qp.token_list == ['(', '(', 'term1', 'AND', 'term2', ')', 'OR', 'term3', ')',
                                     'DIFF', '(', 'term6', 'AND', '(', 'term4', 'OR', 'term5', 
                                     ')', 'DIFF', '(', 'term7', 'OR', 'term8', ')', ')'], "parsing to list salah"
    assert qp.infix_to_postfix() == ['term1', 'term2', 'AND', 'term3', 'OR', 'term6', 'term4',
                                     'term5', 'OR', 'AND', 'term7', 'term8', 'OR', 'DIFF', 'DIFF'], "postfix salah"
    
    qp1 = QueryParser("term1 OR ((term2 AND term3) DIFF (term4 OR term5))", Porter2Stemmer(), set())
    assert qp1.token_list == ['term1', 'OR', '(', '(', 'term2', 'AND', 'term3', ')', 'DIFF', 
                              '(', 'term4', 'OR', 'term5', ')', ')'], "parsing to list salah"
    assert qp1.infix_to_postfix() == ['term1', 'term2', 'term3', 'AND', 'term4', 'term5', 'OR', 
                                      'DIFF', 'OR'], "postfix salah"

    # Testing my Method

    ## Tokenizer
    text = "Hello world! This is a test."
    expected_tokens = ['Hello', 'world', 'This', 'is', 'a', 'test']
    tokens = tokenize_text_by_regex(text)
    assert tokens == expected_tokens, "Tokenization failed"

    ## Stopwords Removal
    tokens = ['this', 'is', 'a', 'simple', 'test']
    stop_words = {'is', 'a'}
    expected_tokens = ['this', 'simple', 'test']
    filtered_tokens = remove_stop_words(tokens, stop_words)
    assert filtered_tokens == expected_tokens, "Stopwords Removal Failed"

    ## Stem Token
    stemmer = LancasterStemmer()
    tokens = ['running', 'jumps', 'easily']
    expected_stemmed_tokens = ['run', 'jump', 'easy']
    stemmed_tokens = stem_tokens(tokens, stemmer)
    assert stemmed_tokens == expected_stemmed_tokens, "Stemming Failed"


