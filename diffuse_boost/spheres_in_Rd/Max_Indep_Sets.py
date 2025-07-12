import numpy as np

import random
from collections import defaultdict

def lubys_algorithm(graph):
    """
    Implements Luby's Algorithm to find a maximal independent set (MIS).

    Parameters:
        graph (dict): Adjacency list representation of the graph. 
                      Example: {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    
    Returns:
        list: A list of nodes in the maximal independent set.
    """
    # Initialize MIS and unmarked vertices
    mis = set()
    unmarked = set(graph.keys())

    while unmarked:
        # Assign random weights to unmarked vertices
        weights = {v: random.random() for v in unmarked}

        # Select vertices to add to MIS
        selected = set()
        for v in unmarked:
            # Check if v has the highest weight among its neighbors
            is_highest = all(weights[v] > weights[neighbor] for neighbor in graph[v] if neighbor in unmarked)
            if is_highest:
                selected.add(v)

        # Add selected vertices to MIS
        mis.update(selected)

        # Remove selected vertices and their neighbors from unmarked set
        to_remove = selected.union(*[set(graph[v]) for v in selected])
        unmarked.difference_update(to_remove)

    return list(mis)

# Example Usage
if __name__ == "__main__":
    # Define a graph using adjacency list
    graph = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 4],
        3: [1, 4],
        4: [2, 3]
    }

    mis = lubys_algorithm(graph)
    print("Maximal Independent Set:", mis)

def MIS_luby(points, min_distance):
    graph = defaultdict(list)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j and np.linalg.norm(p-q) < min_distance:
                graph[i].append(j)
    mis = lubys_algorithm(graph)
    return points[mis].T


def MIS_basic(points, min_distance):
    thinned_points = []
    for p in points:
        closest = 9999999
        for q in thinned_points:
            closest = min(np.linalg.norm(p-q), closest)
        #print("A:", closest, min_distance)
        if closest >= min_distance:
            thinned_points.append(p)
    return np.array(thinned_points).T