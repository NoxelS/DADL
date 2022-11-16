import sys
from utils import *


def _1(path_to_messgae):
    key = remove_duplicates(replace_j_with_i(remove_non_letters([x.upper() for x in sys.argv[1]])))
    text = replace_j_with_i(read_text_from_file(path_to_messgae))
    return key, text

def _2(key):
    rest = [x for x in [chr(x) for x in range(ord("A"), ord("Z")+1)] if (x not in key and x != "J")]
    matrix = [(key+rest)[i*5:i*5+5] for i in range(5)]
    return matrix

def _3(text):
    # "X" wird zwischen zwei gleiche Buchstaben eingefügt
    for i in range(len(text)-1):
        if text[i] == text[i+1]:
            text.insert(i+1, "X")
    
    # "X" wird am Ende des Textes eingefügt, wenn die Anzahl der Buchstaben ungerade ist
    if len(text) % 2 == 1:
        text.append("X")

    bigraph_liste = [(text[i]+text[i+1]) for i in range(0, len(text), 2)]

    return bigraph_liste

def _4(matrix, bigraph_liste):
    return encode_playfair(matrix, bigraph_liste)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python playfair.py <key> <?text-file-path>")
        sys.exit(1)

    path_to_messgae = sys.argv[2] if len(sys.argv) == 3 else "lorem-ipsum.txt"

    # 1.
    key, text = _1(path_to_messgae)
    print(f"key={key}")
    print(f"text={text}")

    # 2.
    matrix = _2(key)
    print(f"matrix={matrix}")

    # 3.
    bigraph_liste = _3(text)
    print(f"bigraph_liste={bigraph_liste}")

    # 4.
    chiffre = _4(matrix, bigraph_liste)
    print(f"chiffre={chiffre}")

    # 5.
    message = decode_playfair(matrix, chiffre)
    print(f"decoded_bigraph_liste={message}")
    message = "".join(message).replace("X", "")
    print(f"decoded_text={message}")