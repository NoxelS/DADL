from tree import Node, BinarySortedNode, RPNNode
import random

if __name__ == "__main__":
    ### Tree Basics
    # Create a tree based on the example in the script
    root = Node('A')
    root.lS = Node('B')
    root.rS = Node('D')
    root.lS.rS = Node('C')
    root.rS.rS = Node('F')
    root.rS.lS = Node('E')
    root.rS.rS.lS = Node('G')

    # Calculate lists
    preOL = root.toPreOrderList()
    infOL = root.toInfixOrderList()
    postOL = root.toPostOrderList()

    # Test lists for correctness
    assert "".join(preOL) == "ABCDEFG"
    assert "".join(infOL) == "BCAEDGF"
    assert "".join(postOL) == "CBEGFDA"

    print("Ordered Lists:")
    print(f"Pre-Order:\t{preOL}")
    print(f"Infix-Order:\t{infOL}")
    print(f"Post-Order:\t{postOL}")

    ### Binary Search Tree
    list = [75, 15, 85, 10, 30, 130, 100, 150,
            115, 140, 160, 120, 145, 170, 190]

    # Create a tree based on a list of numbers
    searchTree =  BinarySortedNode.createTreeFromList(list)
    print(f"\n\nSearched Tree:\nList:\t\t{list}\nTree (infix):\t{searchTree.toInfixOrderList()} ")

    # Finding an element in the tree
    target = random.choice(list)
    element, depth = searchTree.findElement(target)
    print(f"Finding \"{target}\" in the Tree took {depth} compraisons")


    ### Tree based on ordered lists
    preOrderedList = ["A", "B", "C", "D", "E", "F", "G"]
    infixOrderedList = ["B", "C", "A", "E", "D", "G", "F"]
    treeFromOrderedList = Node.createTreeFromOrderedList(preOrderedList.copy(), infixOrderedList.copy())

    # Test if the tree is correct
    postOrderedList = ["C", "B", "E", "G", "F", "D", "A"]
    assert treeFromOrderedList.toPostOrderList() == postOrderedList

    # Print the tree
    print(f"\n\nTree from ordered lists:\nPre-Order:\t{preOrderedList}\nInfix-Order:\t{infixOrderedList}")
    print(f"Tree (post):\t{treeFromOrderedList.toPostOrderList()}")


    ### Reverse Polish Notation
    # Create a tree based on a list of numbers
    rpnTree =  RPNNode.createTreeFromRPNString("61-84/3+2*+")
    mathString = "".join(rpnTree.toMathString())

    # Test if the tree is correct
    assert mathString == "((6-1)+(((8/4)+3)*2))"
    assert rpnTree.calculate() == 15

    print(f"\n\nRPN Tree:\nRPN:\t\t61-84/3+2*+\nTree (infix):\t{rpnTree.toInfixOrderList()} ")
    print(f"Math:\t\t{mathString}")
    print(f"Value:\t\t{rpnTree.calculate()}")

    # Create a tex graph in tikz based on the RPN tree
    file = open('rpn_tree.tex', 'w')
    file.write(rpnTree.toTexString())
    file.close()