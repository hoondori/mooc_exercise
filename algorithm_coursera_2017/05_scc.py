
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

print "original graph", G



# given graph G, given start node
# explore all findable node
def DFS_Explore(G, startNode):
    explored = []
    t = 0 # number of processed node so far


    def DFS(G, node):
        # mark node as explored
        explored.append(node)

        # for all edges from the node, visit all
        for tailNode in G[node]:
            if tailNode not in explored:
                DFS(G,tailNode)

    DFS(G,startNode)

    return explored

# given graph G
# loop over all node, to find scc

def DFS_Loop(G):

    explored = []
    f = defaultdict(int)
    leader = defaultdict(int)

    class local:
        counter = 0

    t = 0 # number of processed node so far
    s = None # current source vertex

    def DFS(G, i):
        #print "visit", i
        # mark node as explored
        explored.append(i)

        # set leader as s
        leader[i] = s

        # for all edges from the node, visit all
        for tailNode in G[i]:
            if tailNode not in explored:
                DFS(G,tailNode)

        # increase finishing time
        local.counter += 1

        # set finishing time for the node
        f[i] = local.counter
        #print "finishing time", i, f[i]

    # assume node label i to n
    for i in range(len(G),0,-1):
        if i not in explored:
            s = i # set current source node
            DFS(G,i)

    return f,leader

#print DFS_Explore(G,4)

# given directed graph, return reversed graph
def Reverse_Graph(G):
    rev_G = defaultdict(list)  # initialize reversed graph

    for v, edges in G.iteritems():
        for w in edges:
            rev_G[w].append(v)

    return rev_G

Grev = Reverse_Graph(G)

f,leader = DFS_Loop(Grev)
print "finishing time", f


# with finishing time, make graph for second-pass
new_G = defaultdict(list)
for i in range(len(G), 0, -1):
    for j in G[i]:
        new_G[f[i]].append(f[j])

#print new_G
f,leader = DFS_Loop(new_G)
print "leader", leader

# find scc with size
scc = defaultdict(list)
for i in range(len(leader), 0, -1):
    scc[leader[i]].append(i)

# print scc with size
for key,value in scc.iteritems():
    print "scc", key, len(value)

# 434821,968,459,313,211