from rhymes import isRhyming

if __name__ == "__main__":
    with open("ATTPT3wortliste.txt", "r", encoding="utf-8") as file, open("rhymes.tmp.txt", "w", encoding="utf-8") as out:
        words = file.read().splitlines()
        print(f"Finding rhymes in {len(words)} words...")
        r=0
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                if isRhyming(words[i], words[j]):
                    out.write(f"{words[i]} - {words[j]}\n")
                    r+=1
        print(f"Found {r} rhymes!")