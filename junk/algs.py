class Node:
    def __init__(self, parent, left, right, val):
        self.parent = parent
        self.left = left
        self.right = right
        self.value = val

def is_bst(root, min, max):
    if root is None:
        return True
    if root.value < min or root.value > max:
        return False
    
    return is_bst(root.left, min, root.value) and is_bst(root.right, root.value, max)

def isBst(root: Node) -> bool:
    return is_bst(root, -10_000, 10_000)


if __name__ == '__main__':
    a = Node(None, None, None, 9)
    b = Node(a, None, None, 7)
    c = Node(a, None, None, 11)
    d = Node(b, None, None, 6)
    e = Node(b, None, None, 8)
    b.left = d
    b.right = e
    a.left, a.right = b, c

    print("isBst(a): ", isBst(a))