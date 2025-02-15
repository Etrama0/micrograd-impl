from graphviz import Digraph

def trace(root):
    # Build a set of all nodes and edges in the graph
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # Left to Right layout

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # For each value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{%s | data: %.4f | grad: %.4f}" % (n.label, n.data, n.grad), shape='record')
        
        if n._op:
            # If the value has an operation, create a node for it
            dot.node(name=uid + n._op, label=n._op)
            # Create an edge from the operation node to the value node
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # Create an edge from n1 to the operation node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
