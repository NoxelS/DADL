def VG(word):
    list = ["" for _ in range(len(word))]
    indizes = [0 for _ in range(len(word))]

    # This algorithm is not fast but it works
    for i, char in enumerate(word):
        if char in "aeiouäöü":
            if list[i-1]:
                list[i-1] += char
                indizes[i-1] = i
            else:
                list[i] += char
                indizes[i] = i

    return ([v for v in [a for a in zip(list, indizes)] if v[0] != ""])

def MVG(word):
    list = VG(word)
    return list[-1] if len(list) > 0 else ["", 0]

def isRhyming(word1, word2):
    rest1 = word1[MVG(word1)[1]:]
    rest2 = word2[MVG(word2)[1]:]

    # 1. Rule: If the last vowel is the same, the words rhyme
    if rest1 != rest2:
        return False

    # 2. Rule:  The MVG and characters after it must contain more than the half of the word
    if (len(rest1) < len(word1)/2) or (len(rest2) < len(word2)/2):
        return False

    # 3. Rule: No word is allowed as suffix in the other word
    if (word1.lower() in word2.lower()) or (word2.lower() in word1.lower()):
        return False

    return True
