class Node:
    def __init__(self, data, lS=None, rS=None,):
        self.lS, self.rS, self.data = lS, rS, data

    def toPreOrderList(self, runner=None):
        runner = [] if runner is None else runner
        runner.append(self.data)
        if self.lS:
            self.lS.toPreOrderList(runner)
        if self.rS:
            self.rS.toPreOrderList(runner)
        return runner

    def toInfixOrderList(self, runner=None):
        runner = [] if runner is None else runner
        if self.lS:
            self.lS.toInfixOrderList(runner)
        runner.append(self.data)
        if self.rS:
            self.rS.toInfixOrderList(runner)
        return runner

    def toPostOrderList(self, runner=None):
        runner = [] if runner is None else runner
        if self.lS:
            self.lS.toPostOrderList(runner)
        if self.rS:
            self.rS.toPostOrderList(runner)
        runner.append(self.data)
        return runner

    ## Generate a tree based on a pre ordered and infix ordered list
    def createTreeFromOrderedList(preList, inList):
        if inList is None or len(inList) == 0:
            return None

        root = Node(preList.pop(0))

        leftIn = inList[:inList.index(root.data)]
        root.lS = Node.createTreeFromOrderedList(preList, leftIn)

        rightIn = inList[inList.index(root.data) + 1:]
        root.rS = Node.createTreeFromOrderedList(preList, rightIn)

        return root

    
    """
        Create a tikz tree from a binary tree
    """
    def createTikzTree(self, body=None, depth=None):
        body = [] if body is None else body
        depth = 0 if depth is None else depth

        ident = "".join(["\t" for _ in range(depth)])

        # First item needs to be the root
        if len(body) == 0:
            body.append(ident + "\\node{" + str(self.data) + "}")
        else:
            body.append(ident + "node{" + str(self.data) + "}")
            
        if self.lS:
            depth+=1
            body.append(ident + "child{")
            self.lS.createTikzTree(body, depth)
            body.append(ident + "}")
        if self.rS:
            if not self.lS:
                depth+=1
            body.append(ident+ "child{")
            self.rS.createTikzTree(body, depth)
            body.append(ident + "}")
        return body
    
    ## Generate a tikz string from a binary tree (this looks like a mess)
    def toTikzString(self):
        return "\\begin{tikzpicture}[\n\tevery node/.style = {minimum width = 1em, draw, circle},\n\tlevel 1/.style ={sibling distance = 3cm},\n\tlevel 2/.style ={sibling distance = 2cm},\n\tlevel 3/.style ={sibling distance = 1cm}]\n" + "\n".join(self.createTikzTree()) + ";\n\\end{tikzpicture}"


    ## Generate a working tex file from a tikz string
    def toTexString(self):
        return "\\documentclass{article}\n\\usepackage{tikz}\n\\begin{document}\n" + self.toTikzString() + "\n\\end{document}"

class BinarySortedNode(Node):
    def __init__(self, data, lS=None, rS=None,):
        super().__init__(data, lS, rS)

    """
        Insert a new node in the tree
    """
    def insertSearchElement(self, value):
        if value > self.data:
            if self.rS:
                self.rS.insertSearchElement(value)
            else:
                self.rS = BinarySortedNode(value)
        elif value <= self.data:
            if self.lS:
                self.lS.insertSearchElement(value)
            else:
                self.lS = BinarySortedNode(value)

    """
        Returns the element if it exists and the number of steps it took to find it
    """
    def findElement(self, target, n=0):
        if self.data == target:
            return (target, n)
        else:
            n += 1
            return self.lS.findElement(target, n) if target <= self.data else self.rS.findElement(target, n)
        return None

    """
        Creates a bineary tree from a list of numbers
    """
    def createTreeFromList(list):
        root = BinarySortedNode(list[0])
        for x in list[1:]:
            root.insertSearchElement(x)
        return root


class RPNNode(Node):
    def __init__(self, data, lS=None, rS=None,):
        super().__init__(data, lS, rS)

    """
        Evaluates the RPN expression and returns the result as float
    """
    def calculate(self):
        if self.data == "+":
            return self.lS.calculate() + self.rS.calculate()
        elif self.data == "-":
            return self.lS.calculate() - self.rS.calculate()
        elif self.data == "*":
            return self.lS.calculate() * self.rS.calculate()
        elif self.data == "/":
            return self.lS.calculate() / self.rS.calculate()
        else:
            return float(self.data)

    """
        Returns the RPN expression as a string with parenthesis
    """
    def toMathString(self, runner=None):
        runner = [] if runner is None else runner
        if self.lS:
            # if self.data in "*/":
            runner.append('(')
            self.lS.toMathString(runner)
        runner.append(self.data)
        if self.rS:
            self.rS.toMathString(runner)
            # if self.data in "*/":
            runner.append(')')
        return runner

    """
        Creates a RPN tree from a reverse polish notation strings
    """
    def createTreeFromRPNString(string):
        operators = ["+", "-", "*", "/"]
        stack = []
        for char in string:
            root = RPNNode(char)
            if char in operators:
                root.rS = stack.pop()
                root.lS = stack.pop()
            stack.append(root)
        return stack.pop()