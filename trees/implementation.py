# from treelib import Node, Tree

# tree = Tree()

# tree.create_node("Harry", "harry")  # No parent means its the root node
# tree.create_node("Jane",  "jane"   , parent="harry")
# tree.create_node("Bill",  "bill"   , parent="harry")
# tree.create_node("Diane", "diane"  , parent="jane")
# tree.create_node("Mary",  "mary"   , parent="diane")
# tree.create_node("Mark",  "mark"   , parent="jane")

# tree.show()

class Node:
    def __init__(self, data, lS=None, rS=None,):
        self.lS, self.rS, self.data = lS, rS, data

    def toPreOrderList(self, runner=[]):
        runner.append(self.data)
        if self.lS:
            self.lS.toPreOrderList(runner)
        if self.rS:
            self.rS.toPreOrderList(runner)
        return runner

    def toInfixOrderList(self, runner=[]):
        if self.lS:
            self.lS.toInfixOrderList(runner)
        runner.append(self.data)
        if self.rS:
            self.rS.toInfixOrderList(runner)
        return runner

    def toPostOrderList(self, runner=[]):
        if self.lS:
            self.lS.toPostOrderList(runner)
        if self.rS:
            self.rS.toPostOrderList(runner)
        runner.append(self.data)
        return runner

    def insertSearchElement(self, value):
        if value > self.data:
            if self.rS:
                self.rS.insertSearchElement(value)
            else:
                self.rS = Node(value)
        elif value <= self.data:
            if self.lS:
                self.lS.insertSearchElement(value)
            else:
                self.lS = Node(value)

    def findElement(self, target, n=0):
        if self.data == target:
            return (target, n)
        else:
            n += 1
            return self.lS.findElement(target, n) if target <= self.data else self.rS.findElement(target, n)


def createSearchTree(list):
    root = Node(list[0])
    for x in list[1:]:
        root.insertSearchElement(x)
    return root


## Pre-In List Generator
def createTreeFromOrderedList(preList, inList):
    if len(inList) == 0:
        return None

    ## This works like visited because the pop also changes the array in a sub sub routine
    root = Node(preList.pop(0))

    leftIn = inList[:inList.index(root.data)]
    root.lS = createTreeFromOrderedList(preList, leftIn)

    rightIn = inList[inList.index(root.data) + 1:]
    root.rS = createTreeFromOrderedList(preList, rightIn)

    return root

# This works
def buildTree(preorder, inorder):

    if not inorder:
        return None
    
    val = preorder.pop(0)
    node = Node(val)
    index = inorder.index(val)
    node.lS = buildTree(preorder,inorder[:index])
    node.rS = buildTree(preorder,inorder[index+1:])

    return node



def calcRPN(list):
    pass


def makeTikzGraph(list):
    pass


if __name__ == "__main__":
    # Build the tree
    # root = Node('A')
    # root.lS = Node('B')
    # root.rS = Node('D')
    # root.lS.rS = Node('C')
    # root.rS.rS = Node('F')
    # root.rS.lS = Node('E')
    # root.rS.rS.lS = Node('G')

    # # Print Lists
    # preOL = root.toPreOrderList()
    # infOL = root.toInfixOrderList()
    # postOL = root.toPostOrderList()

    # print(f"Pre-Order:\t{preOL}")
    # assert "".join(preOL) == "ABCDEFG"
    # print(f"Infix-Order:\t{infOL}")
    # assert "".join(infOL) == "BCAEDGF"
    # print(f"Post-Order:\t{postOL}")
    # assert "".join(postOL) == "CBEGFDA"

    # list = [75, 15, 85, 10, 30, 130, 100, 150,
    #         115, 140, 160, 120, 145, 170, 190]
    # searchTree = createSearchTree(list)
    # target, depth = searchTree.findElement(190)
    # print(f"Found \"{target}\" after {depth} compraisons")
    # searchTreeString = searchTree.toPreOrderList()
    # print(f"Search-Tree (PreOL): {searchTreeString}")

    preO, inO = [1,2,3,4,5],[2,1,4,3,5]
    # n = buildTree(preO, inO)
    n = createTreeFromOrderedList(preO, inO)
    nString = n.toPreOrderList()
    print(f"Tree: {nString}")