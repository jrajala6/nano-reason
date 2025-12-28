from generator import generate_answer, load_model
from verifier import n_attempts
import math
import re
import graphviz

class Node:
    def __init__(self, state, new_content, parent=None, value=0.0, dot=None):
        self.state = state
        self.visits = 0
        self.parent = parent
        self.children = []
        self.new_content = new_content
        self.value = value
        self.is_terminal = re.search(r"\\boxed\{.+?\}", self.state)
        self.dot = dot
        self.id = str(id(self))
        if self.dot:            
            color = "green" if self.value > 0.8 else "red"
            if self.is_terminal: color = "gold"
            
            self.dot.node(
                self.id, 
                label=f"{self.new_content}\n(v={self.value:.2f})", 
                color=color, 
                style="filled", 
                fillcolor=color,
                shape="box"
            )

    def expand(self, model, tokenizer):
        # Don't expand node if it is a terminal node
        if self.is_terminal:
            return self

        if not self.children:
            attempts = n_attempts(model, tokenizer, self.state, max_length=128)
            for candidate_answer, score in attempts:
                self.children.append(Node(state=self.state + "\n" + candidate_answer, new_content=candidate_answer, parent=self, value=score, dot=self.dot))
                if self.dot:
                    self.dot.edge(self.id, self.children[-1].id)
        return self.best_uct_child()

    def best_uct_child(self):
        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return max(unvisited, key=lambda child: child.value)

        return max(self.children, key=lambda child: 
            (child.value / child.visits) + 
            1.41 * math.sqrt(math.log(self.visits) / child.visits)
        )
        
    def backpropagate(self, score, curr_node=False):
        if not curr_node:
            self.value += score
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(score)


def selection_loop(prompt, model, tokenizer, max_iter=10, construct_dot=False):
    dot = None
    if construct_dot:
        dot = graphviz.Digraph(comment='MCTS Search Tree')
        dot.attr(rankdir='TB')

    root = Node(state=prompt, new_content=prompt, dot=dot)
    for _ in range(max_iter):
        node = root
        while node.children:
            node = node.best_uct_child()
        
        best_child = node.expand(model, tokenizer)
        best_child.backpropagate(best_child.value, curr_node=True)
    
    if construct_dot:
        try:
            output_path = dot.render('mcts_tree', format='png', cleanup=True)
            print(f"Tree visualization saved to: {output_path}")
        except Exception as e:
            print(f"Graphviz rendering failed (is graphviz installed?): {e}")
        
    if not root.children:
        return root
    return max(root.children, key=lambda child: child.visits)
    
if __name__ == "__main__":
    model, tokenizer = load_model()
    selection_loop("What is 15 + 13", model, tokenizer, max_iter=5, construct_dot=True)