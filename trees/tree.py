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

    ## Pre-In List Generator
    @staticmethod
    def createTreeFromOrderedList(preList, inList):
        if len(inList) == 0:
            return None

        ## This works like visited because the pop also changes the array in a sub sub routine
        root = Node(preList.pop(0))

        leftIn = inList[:inList.index(root.data)]
        root.lS = Node.createTreeFromOrderedList(preList, leftIn)

        rightIn = inList[inList.index(root.data) + 1:]
        root.rS = Node.createTreeFromOrderedList(preList, rightIn)

        return root


class BinarySortedNode(Node):
    def __init__(self, data, lS=None, rS=None,):
        super().__init__(data, lS, rS)

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

    def findElement(self, target, n=0):
        if self.data == target:
            return (target, n)
        else:
            n += 1
            return self.lS.findElement(target, n) if target <= self.data else self.rS.findElement(target, n)

    @staticmethod
    def createTreeFromList(list):
        root = BinarySortedNode(list[0])
        for x in list[1:]:
            root.insertSearchElement(x)
        return root

class RPNNode(Node):
    def __init__(self, data, lS=None, rS=None,):
        super().__init__(data, lS, rS)

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

    def toMathString(self, runner=None):
        runner = [] if runner is None else runner
        if self.lS:
            if self.data in "*/":
                runner.append('(')
            self.lS.toMathString(runner)
        runner.append(self.data)
        if self.rS:
            self.rS.toMathString(runner)
            if self.data in "*/":
                runner.append(')')
        return runner

    @staticmethod
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