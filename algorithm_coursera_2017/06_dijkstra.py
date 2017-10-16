import csv as csv
import random as random
import copy as copy
from collections import defaultdict

#f = open('./dijkstraTest.txt','r')
f = open('./dijkstraData.txt','r')
lines = f.readlines()


G = defaultdict(list) # graph
for line in lines:

    splits = line.split()

    # head node label
    v = int(splits[0])

    for s in splits[1:]:
        a = s.split(',')
        w = int(a[0]) # tail node label
        cost = int(a[1]) # edge cost

        # append tail node to head node's edge list
        G[v].append((w,cost))

print "original graph", G


def dijkstra(G, s):
    X = [s] # processed nodes so far
    A = defaultdict(int)
    B = defaultdict(list)
    A[s] = 0 # computed shortest path distances for source s
    B = [s] # empty path

    while len(X) <= len(G):
        # need to grow X by one node
        # among all edges (v,w),  v node in X, w node not in X
        # find the edge that minimize A[v] + lvw
        candidates = []
        for v in X:
            edges = G[v]
            for (w,cost) in edges:
                if w not in X:
                    candidates.append( (v,w,A[v]+cost) )   # v,w,score
        print "candidates", candidates

        # find minimum path from candidates
        min_candidate = (None,None,99999999)
        for candidate in candidates:
            if( min_candidate[2] > candidate[2]):
                min_candidate = candidate
        print "min path", min_candidate

        # include minimum path to X, and calcuate A,B
        X.append(min_candidate[1])
        A[min_candidate[1]] = min_candidate[2]
        B.append(min_candidate[1])

    return A,B

A,B = dijkstra(G,1)
print A
print B
print A[7],A[37],A[59],A[82],A[99],A[115],A[133],A[165],A[188],A[197]