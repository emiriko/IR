import re
from typing import List, Optional

from porter2stemmer import Porter2Stemmer


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

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        # TODO (FINISHED)
        if type(key) == str:
            return self.__get_id(key)
        else:
            return self.__get_str(key)


def merge_and_sort_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparable, int)]
        Penggabungan yang sudah terurut
    """
    # TODO (FINISHED)

    term_frequencies = dict()

    for post in posts_tfs1:
        term_frequencies[post[0]] = post[1]

    for post in posts_tfs2:
        if post[0] in term_frequencies:
            term_frequencies[post[0]] = term_frequencies[post[0]] + post[1]
        else:
            term_frequencies[post[0]] = post[1]

    result = []

    sorted_term_frequencies = dict(sorted(term_frequencies.items()))

    for docID, frequency in sorted_term_frequencies.items():
        result.append((docID, frequency))

    return result


def tokenize_text_by_regex(text: str, pattern: Optional[str] = r'\b\w+\b'):
    """
    Tokenisasi teks berdasarkan pola tokenizer_pattern
    """
    text_lowercase = text.lower()

    tokens = re.findall(pattern, text_lowercase)

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

    assert [term_id_map[term]
            for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname]
            for docname in docs] == [0, 1, 2], "docs_id salah"

    assert merge_and_sort_posts_and_tfs([(1, 34), (3, 2), (4, 23)],
                                        [(1, 11), (2, 4), (4, 3), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "merge_and_sort_posts_and_tfs salah"

    # Testing my Method

    ## Tokenizer
    text = "Hello world! This is a test."
    expected_tokens = ['hello', 'world', 'this', 'is', 'a', 'test']
    tokens = tokenize_text_by_regex(text)
    assert tokens == expected_tokens, "Tokenization failed"

    ## Stopwords Removal
    tokens = ['this', 'is', 'a', 'simple', 'test']
    stop_words = {'is', 'a'}
    expected_tokens = ['this', 'simple', 'test']
    filtered_tokens = remove_stop_words(tokens, stop_words)
    assert filtered_tokens == expected_tokens, "Stopwords Removal Failed"

    ## Stem Token
    stemmer = Porter2Stemmer()
    tokens = ['runner', 'eat']
    expected_stemmed_tokens = ['runner', 'eat']
    stemmed_tokens = stem_tokens(tokens, stemmer)
    assert stemmed_tokens == expected_stemmed_tokens, "Stemming Failed"