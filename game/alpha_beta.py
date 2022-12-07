#!/usr/bin/env python3
from math import inf

def val(node):
    return node.val

def minimax(node, depth, alpha, beta, is_max):
    # We are at the leaf node
    if not depth:
        return val(node)
    if is_max:
        max_val = -inf
        for i in node.children:
            # If parent is max then the children will be min
            tmp = minimax(i, depth-1, alpha, beta, is_max=False)
            max_val = max(tmp, max_val)
            alpha = max(tmp, alpha)
            # Beta -> lowest value currently found
            # Alpha -> Greatest value currently found
            # We are in a MAX node. MAX choses the largest value possible.
            # That means it's parent is a MIN which choses the smallest
            # If there is a node smaller than the current largest i.e the one
            # that may be chosen by MAX then MIN will never choose this.
            # So no need to search it.
            if beta <= alpha:
                break
        return max_val
    else:
        min_val = inf
        for i in node.children:
            # If parent is min then the children will be max
            tmp = minimax(i, depth-1, alpha, beta, is_max=True)
            min_val = min(tmp, min_val)
            beta = min(beta, tmp)
            # Similar to the above situation, the parent will be a MIN
            # If there is value that is larger than the current beta that means
            # this branch will never be chosen
            # Why does it work?
            # MIN choses the minimum possible value
            # The chosen value is lower or equal to beta
            # So the maximum that this node can get is beta.
            # If a value exists that is greater then MAX will always chose that
            if beta <= alpha:
                break
        return min_val

