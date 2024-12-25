import array
import struct
from typing import List

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()


class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def encode(postings_list: List[int]):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """

        # TODO(FINISHED)

        list_of_gaps = []
        prev = -1
        for idx in range(len(postings_list)):
            document = postings_list[idx]
            if idx == 0:
                list_of_gaps.append(document)
            else:
                list_of_gaps.append(document - prev)
            prev = document

        return VBEPostings.vb_encode(list_of_gaps)
    
    @staticmethod
    def vb_encode(list_of_numbers: List[int]):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        # TODO (FINISHED)
        bytestream = bytearray()

        for n in list_of_numbers:
            result = VBEPostings.vb_encode_number(n)
            bytestream.extend(result)

        return bytestream

    @staticmethod
    def vb_encode_number(number: int):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        # TODO (FINISHED)

        bytes_list = []

        while True:
            bytes_list.insert(0, number % 128)
            if number < 128:
                break
            number = number // 128

        bytes_list[-1] += 128

        return bytes_list

    @staticmethod
    def decode(encoded_postings_list: bytes):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        # TODO (FINISHED)

        gaps_decoded = VBEPostings.vb_decode(encoded_postings_list)
        prev = -1
        list_of_numbers = []

        for idx in range(len(gaps_decoded)):
            document = gaps_decoded[idx]
            if idx == 0:
                list_of_numbers.append(document)
                prev = document
            else:
                list_of_numbers.append(document + prev)
                prev = document + prev

        return list_of_numbers

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        numbers = []
        n = 0
        for byte in encoded_bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + byte - 128
                numbers.append(n)
                n = 0
        return numbers


class EliasGammaPostings:
    @staticmethod
    def encode(postings_list: List[int]) -> bytes:
        list_of_gaps = []
        prev = 0
        for idx, docID in enumerate(postings_list):
            if idx == 0:
                list_of_gaps.append(docID)
            else:
                list_of_gaps.append(docID - prev)
            prev = docID

        bitstream = EliasGammaPostings.eg_encode(list_of_gaps)

        return EliasGammaPostings.bitstream_to_bytes(bitstream)

    @staticmethod
    def eg_encode(list_of_numbers: List[int]) -> str:
        bitstream = ""
        for n in list_of_numbers:
            encoded_number = EliasGammaPostings.eg_encode_number(n)
            bitstream += encoded_number
        return bitstream

    @staticmethod
    def eg_encode_number(number: int) -> str:
        binary_rep = bin(number)[2:]
        n = len(binary_rep) - 1
        unary_part = '0' * n + '1'
        gamma_code = unary_part + binary_rep[1:]
        return gamma_code

    @staticmethod
    def bitstream_to_bytes(bitstream: str) -> bytes:
        padding_length = (8 - len(bitstream) % 8) % 8
        bitstream = bitstream + '0' * padding_length

        bitstream_int = int(bitstream, 2)

        num_bytes = (len(bitstream) + 7) // 8
        byte_data = bitstream_int.to_bytes(num_bytes, byteorder='big')

        length_prefix = struct.pack(">I", len(byte_data))
        return length_prefix + byte_data

    @staticmethod
    def decode(encoded_bytes: bytes) -> List[int]:
        if len(encoded_bytes) < 4:
            raise ValueError("Encoded data is too short to contain a valid length prefix.")

        length_prefix = encoded_bytes[:4]
        data_length = struct.unpack(">I", length_prefix)[0]

        byte_data = encoded_bytes[4:4 + data_length]

        if len(byte_data) != data_length:
            raise ValueError("The encoded data does not match the expected length.")

        bitstream = EliasGammaPostings.bytes_to_bitstream(byte_data, data_length * 8)

        gaps_decoded = EliasGammaPostings.eg_decode(bitstream)

        prev = -1
        list_of_numbers = []

        for idx, document in enumerate(gaps_decoded):
            if idx == 0:
                list_of_numbers.append(document)
                prev = document
            else:
                list_of_numbers.append(document + prev)
                prev = document + prev

        return list_of_numbers

    @staticmethod
    def bytes_to_bitstream(encoded_bytes: bytes, valid_bit_length: int) -> str:
        bitstream_int = int.from_bytes(encoded_bytes, byteorder='big')

        bitstream = bin(bitstream_int)[2:]

        bitstream = bitstream.zfill(len(encoded_bytes) * 8)

        return bitstream[:valid_bit_length]

    @staticmethod
    def eg_decode(encoded_bitstream: str) -> List[int]:
        decoded_numbers = []
        i = 0

        while i < len(encoded_bitstream):
            n = 0
            while i < len(encoded_bitstream) and encoded_bitstream[i] == '0':
                n += 1
                i += 1

            if i >= len(encoded_bitstream):
                break

            i += 1

            if n > 0:
                if i + n > len(encoded_bitstream):
                    break
                binary_part = '1' + encoded_bitstream[i:i + n]
                i += n
            else:
                binary_part = '1'

            decoded_number = int(binary_part, 2)
            decoded_numbers.append(decoded_number)

        return decoded_numbers

if __name__ == '__main__':
    postings_list = [34, 67, 89, 454, 2345738]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)

        print(encoded_postings_list)
        print("byte hasil encode: ", encoded_postings_list)
        print("ukuran encoded postings: ", len(encoded_postings_list), "bytes")
        decoded_posting_list = Postings.decode(encoded_postings_list)
        print("hasil decoding: ", decoded_posting_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        print()
