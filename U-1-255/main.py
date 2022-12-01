from utils import timeit, output_timeit, catalan
from algorithms import PrintPerm, PermIter
from tree import BinarySortedNode

@timeit
def Exercise2():
    print("Exercise 2:")
    print("All permutations of A = [1,2,3,4]")

    A = [1,2,3,4]
    PrintPerm(A, len(A))

    print("All permutations of A = [1,2,3,4] with an iterator")

    A = [1,2,3,4]
    for p in PermIter(A, len(A)):
        print(p)

@timeit
def Exercise3():
    print("Exercise 3:")
    print("GenerateSortTree(list)")
    for p in PermIter([1,2,3,4], 4):
        n = BinarySortedNode.create_tree_from_list(p)
        print(n.to_pre_order_list())


@timeit
def Exercise4():
    print("Exercise 4:")
    print("Print UCode")
    for p in PermIter([1,2,3,4], 4):
        n = BinarySortedNode.create_tree_from_list(p)
        print("".join(n.generate_ucode()))

@timeit
def Exercise5():
    print("Exercise 5:")
    print("Dictionary for topological classes")

    d = {}
    for p in PermIter([1,2,3,4], 4):
        n = BinarySortedNode.create_tree_from_list(p)
        ucode = "".join(n.generate_ucode())
        ucodePre = "".join(n.generate_ucode_mirrored())
        if ucode in d:
            d[ucode] += 1
        else:
            d[ucode] = 1

    print(d)

@timeit
def Exercise6():
    print("Exercise 6:")
    print("Dictionary for topological classes for mirrored trees")

    d = {}
    for p in PermIter([1,2,3,4], 4):
        n = BinarySortedNode.create_tree_from_list(p)
        ucode = "".join(n.generate_ucode())
        ucodePre = "".join(n.generate_ucode_mirrored())
        
        if ucode in d:
            d[ucode] += 1
        else:
            if ucodePre in d:
                d[ucodePre] = 1
            else:
                d[ucode] = 1

    print(d)

@timeit
def Exercise7():
    print("Exercise 7:")
    print("Make table of the first topological classes")

    for n in range(1, 7):
        A = [i for i in range(1,n+1)]
        d = {}
        print(A)
        for p in PermIter(A, n):
            node = BinarySortedNode.create_tree_from_list(p)
            ucode = "".join(node.generate_ucode())
            if ucode not in d:
                d[ucode] = node

        # Write everything to a beautiful table, this is very hacky and ugly but it works pretty well
        with open(str(n) + ".topologies.tmp.tex", "w") as f:
            f.write("\\documentclass{article}\\usepackage{tikz}\\usepackage{pbox}\\begin{document}\\hoffset=-1in\\voffset=-1in\\setbox0\\hbox{\\begin{tabular}{ |c|c|c|c|c|c| }")
            f.write("\\hline\n")
            for i, k in enumerate(d):
                f.write("\\pbox{20cm}{[" + k +"] \\\\ \\\\")
                f.write(d[k].to_tikz_topology())
                f.write("\\\\ }")
                f.write(" & " if (i + 1) % 6 != 0 else " \\\\ \\hline \n")
            f.write("\\\\ \\hline\n")
            f.write("\\end{tabular}}\\pdfpageheight=\\dimexpr\\ht0+\\dp0\\relax\\pdfpagewidth=\\wd0\\shipout\\box0\\stop")

if __name__ == "__main__":
    Exercise2() 
    Exercise3() 
    Exercise4()
    Exercise5()
    Exercise6()
    Exercise7()

    output_timeit()