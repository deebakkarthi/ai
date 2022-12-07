#!/usr/bin/env python3
from math import inf

def val(node):
    return node.val

def minimax(node, depth, is_max):
    # We are at the leaf node
    if not depth:
        return val(node)
    if is_max:
        max_val = -inf
        for i in node.children:
            # If parent is max then the children will be min
            tmp = minimax(i, depth-1, is_max=False)
            max_val = max(tmp, max_val)
        return max_val
    else:
        min_val = inf
        for i in node.children:
            # If parent is min then the children will be max
            tmp = minimax(i, depth-1, is_max=True)
            min_val = min(tmp, min_val)
        return min_val

