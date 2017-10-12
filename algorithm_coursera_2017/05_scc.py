
import csv as csv
import random as random
import copy as copy
from collections import defaultdict

f = open('./scc_small.txt');
reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)

G = defaultdict(list) # graph
for row in reader:

    # head node
    v = int(row[0])

    # tail node
    w = int(row[1])

    # append tail node to head node's edge list
    G[v].append(w)

print G



# given graph G, given start node
# explore all findable node
def DFS_Explore(G, startNode):
    explored = []

    def DFS(G, node):
        # mark node as explored
        explored.append(node)

        # for all edges from the node, visit all
        for tailNode in G[node]:
            if tailNode not in explored:
                DFS(G,tailNode)

    DFS(G,startNode)

    return explored

print DFS_Explore(G,4)

# given directed graph, return reversed graph
def Reverse_Graph(G):
    rev_G = defaultdict(list)  # initialize reversed graph

    for v, edges in G.iteritems():
        for w in edges:
            rev_G[w].append(v)

    return rev_G

print Reverse_Graph(G)

