class Node:
    def __init__(self, data, lS=None, rS=None,):
        self.lS, self.rS, self.data = lS, rS, data

    def insert(self, ldata, rdata):
        if self.lS or self.rS:
            raise Exception("Cannot insert into a node with children")
        self.lS = Node(ldata)
        self.rS = Node(rdata)
        return self.lS, self.rS

    ## We could also use travel and visit methods to generate the list but this is compact
    def to_pre_order_list(self, runner=None):
        runner = [] if runner is None else runner
        runner.append(self.data)
        if self.lS:
            self.lS.to_pre_order_list(runner)
        if self.rS:
            self.rS.to_pre_order_list(runner)
        return runner

    ## We could also use travel and visit methods to generate the list but this is compact
    def to_infix_order_list(self, runner=None):
        runner = [] if runner is None else runner
        if self.lS:
            self.lS.to_infix_order_list(runner)
        runner.append(self.data)
        if self.rS:
            self.rS.to_infix_order_list(runner)
        return runner

    ## We could also use travel and visit methods to generate the list but this is compact
    def to_post_order_list(self, runner=None):
        runner = [] if runner is None else runner
        if self.lS:
            self.lS.to_post_order_list(runner)
        if self.rS:
            self.rS.to_post_order_list(runner)
        runner.append(self.data)
        return runner

    @staticmethod
    def create_tree_from_pre_in(pre_list, in_list):
        if in_list is None or len(in_list) == 0:
            return None

        root = Node(pre_list.pop(0))

        left_in = in_list[:in_list.index(root.data)]
        root.lS = Node.create_tree_from_pre_in(pre_list, left_in)

        right_in = in_list[in_list.index(root.data) + 1:]
        root.rS = Node.create_tree_from_pre_in(pre_list, right_in)

        return root

    @staticmethod
    def create_tree_from_in_post(in_list, post_list):
        if in_list is None or len(in_list) == 0:
            return None

        root = Node(post_list.pop())

        right_in = in_list[in_list.index(root.data) + 1:]
        root.rS = Node.create_tree_from_in_post(right_in, post_list)

        left_in = in_list[:in_list.index(root.data)]
        root.lS = Node.create_tree_from_in_post(left_in, post_list)

        return root

    @staticmethod
    def create_tree_from_pre_post(pre_list, post_list):
        # Currently not working
        return Node(None)

    def print_tree(self, level=0):
        if self.rS:
            self.rS.print_tree(level + 1)
        print(' ' * 5 * level + str(self.data))
        if self.lS:
            self.lS.print_tree(level + 1)

    def print_tree_pre_order(self):
        for i,s in enumerate(self.to_pre_order_list()):
            print(str(s), end="\n" if (i+1) % 3 == 0 else " ")
        
        # End line if not already ended
        if len(self.to_pre_order_list()) % 3 != 0:
            print()

    def __str__(self):
        return str(self.data or "-")
    
    def depth(self):
        return max(self.lS.depth() if self.lS else 0, self.rS.depth() if self.rS else 0) + 1

    def visit(self, value):
        print(f"Visiting {self}")
        return self if self.data == value else None

    def find_element_pre_ordered(self, target_data):
        if self:
            return self.visit(target_data) or \
            (self.lS.find_element_pre_ordered(target_data) if self.lS else None) or \
            (self.rS.find_element_pre_ordered(target_data) if self.rS else None)

    def find_element_post_ordered(self, target_data):
        if self:
            return (self.lS.find_element_post_ordered(target_data) if self.lS else None) or \
            (self.rS.find_element_post_ordered(target_data) if self.rS else None) or \
            self.visit(target_data)

    def find_element_infix_ordered(self, target_data):
        if self:
            return (self.lS.find_element_infix_ordered(target_data) if self.lS else None) or \
            self.visit(target_data) or \
            (self.rS.find_element_infix_ordered(target_data) if self.rS else None)

    def create_tikz_tree(self, body=None, depth=None):
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
            self.lS.create_tikz_tree(body, depth)
            body.append(ident + "}")
        if self.rS:
            if not self.lS:
                depth+=1
            body.append(ident+ "child{")
            self.rS.create_tikz_tree(body, depth)
            body.append(ident + "}")
        return body

    def create_tikz_tree_topology(self, body=None, depth=None):
        body = [] if body is None else body
        depth = 0 if depth is None else depth

        ident = "".join(["\t" for _ in range(depth)])

        # First item needs to be the root
        if len(body) == 0:
            body.append(ident + "\\node{}")
        else:
            body.append(ident + "node{}")
            
        if self.lS:
            if self.lS.data != "x":
                depth+=1
                body.append(ident + "child{")
                self.lS.create_tikz_tree_topology(body, depth)
                body.append(ident + "}")
            else:
                body.append(ident + "child [missing]")
        if self.rS:
            if self.rS.data != "x":
                if not self.lS:
                    depth+=1
                body.append(ident+ "child{")
                self.rS.create_tikz_tree_topology(body, depth)
                body.append(ident + "}")
            else:
                body.append(ident + "child [missing]")
        return body

    def to_tikz_string(self):
        return "\\begin{tikzpicture}[\n\tevery node/.style = {minimum width = 1em, draw, circle},\n\tlevel 1/.style ={sibling distance = 3cm},\n\tlevel 2/.style ={sibling distance = 2cm},\n\tlevel 3/.style ={sibling distance = 1cm}]\n" + "\n".join(self.create_tikz_tree()) + ";\n\\end{tikzpicture}"

    def to_tikz_topology(self):
        return "\\begin{tikzpicture}[every node/.style = {minimum width = .01em, draw, circle}]\\tikzset{level distance=10mm,level/.style={sibling distance="+ str(self.depth() * 10) + "mm/#1}}" + "\n".join(self.create_tikz_tree_topology()) \
             + ";\n\\end{tikzpicture}"
            # + ";\\node[xshift=3ex, yshift=-2ex, overlay, fill=none, draw=white, above right] at (current bounding box.north west) { \\textbf{" + "".join(self.generate_ucode()) + "}};" \

    def to_tex_string(self):
        return "\\documentclass{article}\n\\usepackage{tikz}\n\\begin{document}\n" + self.to_tikz_string() + "\n\\end{document}"


    def fill_holes(self):
        if self.lS or self.rS:
            if self.lS is None:
                self.lS = Node("x")
            if self.rS is None:
                self.rS = Node("x")
            self.lS.fill_holes()
            self.rS.fill_holes()

    def generate_ucode(self, runner=None):
        self.fill_holes()

        runner = [] if runner is None else runner

        if self.lS:
            self.lS.generate_ucode(runner)
        if self.rS:
            self.rS.generate_ucode(runner)
        
        if self.lS or self.rS:
            runner.append("n")
        else:
            runner.append("x" if self.data == "x" else "b")

        return [str(_) for _ in runner]

    def generate_ucode_mirrored(self, runner=None):
        self.fill_holes()

        runner = [] if runner is None else runner

        if self.lS or self.rS:
            runner.append("n")
        else:
            runner.append("x" if self.data == "x" else "b")

        if self.lS:
            self.lS.generate_ucode_mirrored(runner)
        if self.rS:
            self.rS.generate_ucode_mirrored(runner)
        

        return list(reversed([str(_) for _ in runner]))

class BinarySortedNode(Node):
    def __init__(self, data, lS=None, rS=None,):
        super().__init__(data, lS, rS)

    def insert_search_element(self, value):
        # If the value is less than the current node it goes to the left
        if value > self.data:
            # Traverse recursively if there is a right node
            if self.rS:
                self.rS.insert_search_element(value)
            else:
                self.rS = BinarySortedNode(value)
        # If the value is greater than the current node it goes to the right
        elif value <= self.data:
            # Traverse recursively if there is a left node
            if self.lS:
                self.lS.insert_search_element(value)
            else:
                self.lS = BinarySortedNode(value)

    def find_element(self, target_data, n=0):
        if self.data == target_data:
            return (self, n)
        # Search recursively on the left subtree if the target is less than the current node and wise versa for the right subtree
        else:
            n += 1
            return self.lS.find_element(target_data, n) if target_data <= self.data else self.rS.find_element(target_data, n)

    def find_min(self):
        # Min is the leftmost node in the tree
        if self.lS:
            return self.lS.find_min()
        return self

    def remove_element(self, target_data):
        # If the target is found, we need to edit the children
        if self.data == target_data:
            # No children: Just remove the node
            if not self.lS and not self.rS:
                return None
            # One child: Replace the node with the only child
            elif not self.lS or not self.rS:
                return self.lS if self.lS else self.rS
            # Two children: Replace the node with the minimum of the right subtree
            else:
                # Find the min of the right sub tree
                min_node = self.rS.find_min()
                self.data = min_node.data
                self.rS = self.rS.remove_element(min_node.data)
                return self
        # If the target is less than the current node, search the left subtree
        elif target_data < self.data:
            self.lS = self.lS.remove_element(target_data)
        # If the target is greater than the current node, search the right subtree
        else:
            self.rS = self.rS.remove_element(target_data)
        return self

    @staticmethod
    def create_tree_from_list(list):
        root = BinarySortedNode(list[0])
        # Just insert the rest of the list
        for x in list[1:]:
            root.insert_search_element(x)
        return root

    @staticmethod
    def create_balanced_tree_from_list(list):
        if len(list) == 0:
            return None
        else:
            # Sort the list
            list = sorted(list)
            # The middle of the list is the root of the tree and the left and right parts are the left and right subtrees
            mid = len(list) // 2
            root = BinarySortedNode(list[mid])
            # Recursively create the left and right subtrees
            root.lS = BinarySortedNode.create_balanced_tree_from_list(list[:mid])
            root.rS = BinarySortedNode.create_balanced_tree_from_list(list[mid+1:])
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

    """
        Returns the RPN expression as a string with parenthesis
    """
    def to_math_string(self):
        return "".join(self.to_math_string_helper()[1:-1])

    def to_math_string_helper(self, runner=None):
        runner = [] if runner is None else runner
        if self.lS:
            runner.append('(')
            self.lS.to_math_string_helper(runner)
        runner.append(self.data)
        if self.rS:
            self.rS.to_math_string_helper(runner)
            runner.append(')')
        return runner

    @staticmethod
    def create_tree_from_RPN_string(string):
        operators = ["+", "-", "*", "/"]
        stack = []
        for char in string:
            root = RPNNode(char)
            if char in operators:
                root.rS = stack.pop()
                root.lS = stack.pop()
            stack.append(root)
        return stack.pop()