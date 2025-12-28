from .generator import generate_answer
from .verifier import n_attempts
import math
import re

class Node:
    def __init__(self, state, parent=None, value=0.0):
        self.state = state
        self.visits = 0
        self.parent = parent
        self.children = []
        self.value = value
        self.is_terminal = re.search(r"\\boxed\{.+?\}", self.state)

    def expand(self, model, tokenizer):
        # Don't expand node if it is a terminal node
        if self.is_terminal:
            return self

        if not self.children:
            attempts = n_attempts(model, tokenizer, self.state, max_length=128)
            for candidate_answer, score in attempts:
                self.children.append(Node(state=self.state + "\n" + candidate_answer, parent=self, value=score))
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


def selection_loop(prompt, model, tokenizer, max_iter=10):
    root = Node(state=prompt)
    for _ in range(max_iter):
        node = root
        while node.children:
            node = node.best_uct_child()
        
        if node.is_terminal:
            return node
        
        best_child = node.expand(model, tokenizer)
        best_child.backpropagate(best_child.value, curr_node=True)
    return max(root.children, key=lambda child: child.visits)
    