
def remove_non_letters(text):
    return [x for x in text if x in [chr(x) for x in range(ord("A"), ord("Z")+1)]]

def read_text_from_file(path_to_messgae):
    text = ""
    with open(path_to_messgae, "r", encoding="utf-8") as file:
        text = remove_non_letters([x.upper() for x in file.read()])
    return text

def replace_j_with_i(text):
    return [x if x != "J" else "I" for x in text]

def remove_duplicates(text):
    return list(dict.fromkeys(text))

def get_coordinates(matrix, letter):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == letter:
                return i, j

def playfair_algorithus(matrix, bigraph_liste, reverse=False):
    chiffre = ["" for _ in bigraph_liste]
    for bigraph_index, bigraph in enumerate(bigraph_liste):
        x1, y1 = get_coordinates(matrix, bigraph[0])
        x2, y2 = get_coordinates(matrix, bigraph[1])

        # (a) Beide Buchstaben liegen in derselben Zeile
        if y1 == y2:
            # Jeder Buchstabe wird jeweils durch den rechten Nachbarn ersetzt (periodische Ränder)
            x1 = (x1 + (- 1 if reverse else 1)) % 5
            x2 = (x2 + (- 1 if reverse else 1)) % 5

        # (b) Beide Buchstaben liegen in derselben Spalte
        elif x1 == x2:
            # Jeder Buchstabe wird jeweils durch den unteren Nachbarn ersetzt (periodische Ränder)
            y1 = (y1 + (- 1 if reverse else 1)) % 5
            y2 = (y2 + (- 1 if reverse else 1)) % 5

        # (c) Beide Buchstaben bilden zwei Ecken eines gedachten Quadrats innerhalb der Matrix
        else:
            # Jeder Buchstabe wird jeweils durch einen derbeiden andren "Eckbuchstaben" ersetzt
            y1, y2 = y2, y1
        chiffre[bigraph_index] = matrix[x1][y1] + matrix[x2][y2]

    return chiffre

def encode_playfair(matrix, bigraph_liste):
    return playfair_algorithus(matrix, bigraph_liste)

def decode_playfair(matrix, chiffre):
    return playfair_algorithus(matrix, chiffre, reverse=True)