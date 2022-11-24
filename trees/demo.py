from tree import Node, BinarySortedNode, RPNNode

def hline(title="", char='=', length=80):
    n = (length - len(title))//2
    print(char * n, title, char * n,)

def wrap(title, func, *args):
    hline(title)
    func(*args)

if __name__ == "__main__":
    # a) - c)
    rootA = Node(59)
    a,b = rootA.insert(31,67)
    c,d = a.insert(12,39)
    e,f = b.insert(7,22)
    g,h = c.insert(35,43)
    i,j = d.insert(62,93)
    wrap("10. a)-c) Test Tree", rootA.print_tree)
    
    # d)
    hline("10. d) Traversal")
    print("pre_order_list = ", rootA.to_pre_order_list())
    print("infix_order_list = ", rootA.to_infix_order_list())
    print("post_order_list = ", rootA.to_post_order_list())

    # e)
    hline("10. e) PrintTree(node)")
    rootA.print_tree_pre_order()

    # 11.
    hline("11. Searching for 39 in the tree")
    hline("Pre-ordered Search", char='.')
    print("Found:", rootA.find_element_pre_ordered(39))
    hline("Post-ordered Search", char='.')
    print("Found:", rootA.find_element_post_ordered(39))
    hline("Infix-ordered Search", char='.')
    print("Found:", rootA.find_element_infix_ordered(39))

    # 12.
    hline("12. Create Tree from ordered list")
    preList = rootA.to_pre_order_list()
    infixList = rootA.to_infix_order_list()
    postList = rootA.to_post_order_list()
    print("Pre-ordered list:", preList)
    print("Infix-ordered list:", infixList)
    print("Post-ordered list:", postList)

    hline("From pre and infix", char='.')
    rootB = Node.create_tree_from_pre_in(preList, infixList)
    print("Tree:")
    rootB.print_tree()
    print("Pre-ordered list:", rootB.to_pre_order_list())

    hline("From infix and post", char='.')
    rootC = Node.create_tree_from_in_post(infixList, postList)
    print("Tree:")
    rootC.print_tree()
    print("Post-ordered list:", rootC.to_post_order_list())

    hline("From pre and post", char='.')
    rootD = Node.create_tree_from_pre_post(preList, postList)
    print("Tree:")
    rootD.print_tree()
    print("Pre-ordered list:", rootD.to_pre_order_list())

    # 13.
    hline("13. Create sort tree from unordered list")
    unorderedList = [59, 31, 67, 12, 39, 7, 22, 35, 43, 62, 93]
    rootE = BinarySortedNode.create_tree_from_list(unorderedList)
    print("Unordered list:", unorderedList)
    print("Tree:")
    rootE.print_tree()

    # 14.
    hline("14. Remove element from sorted tree")
    print("Tree:")
    rootE.print_tree()
    print("Remove 31")
    rootE.remove_element(31)
    print("Tree:")
    rootE.print_tree()

    # 15.
    hline("15. Create balanced tree from unordered list")
    unorderedList = [59, 31, 67, 12, 39, 7, 22, 35, 43, 62, 93, 1, 23, 2, 5]
    rootF = BinarySortedNode.create_balanced_tree_from_list(unorderedList)
    print("Unordered list:", unorderedList)
    print("Tree:")
    rootF.print_tree()

    # 16.
    hline("16. Create RPN tree from RPN list")
    rpnString = "61-84/3+2*+"
    rootG = RPNNode.create_tree_from_RPN_string(rpnString)
    print("RPN string:", rpnString)
    print("Tree:")
    rootG.print_tree()
    print("Result:", rootG.calculate())
    print("Math string:", "".join(rootG.to_math_string()))

    #17. 
    hline("17. Generate Tikz Tree")
    tikzString = rootA.to_tikz_string()
    print("Tree:")
    rootA.print_tree()
    print("Tikz Tree:")
    print(tikzString)
    print("Tikz Tree in file 'tikzTree.tmp.tex'")
    with open("tikzTree.tmp.tex", "w") as f:
        f.write(rootA.to_tex_string())