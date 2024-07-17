import nn
from engine import Value
import graph

def trace(root):
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
    dot = graph.Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    nodes, edges = trace(root)
    for n in nodes:
        dot.node(name=str(id(n)), label=f"data: {n.data:.4f}\\ngrad: {n.grad:.4f}", shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

# Example usage
if __name__ == "__main__":
    # Create a simple neural network
    model = nn.MLP(2, [4, 4, 1])
    
    # Create input data
    X = [[Value(1.0), Value(-2.0)], [Value(-1.0), Value(2.0)]]
    y = [Value(1.0), Value(-1.0)]
    
    # Train the model
    nn.train(model, X, y, learning_rate=0.1, epochs=100)
    
    # Make a prediction
    x_test = [Value(0.5), Value(-1.0)]
    y_pred = model.forward(x_test)
    
    # Visualize the computation graph
    dot = draw_dot(y_pred)
    dot.render('computation_graph', view=True)
