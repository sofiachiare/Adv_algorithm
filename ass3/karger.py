import math
import heapq
from pickle import NONE
import random
import time
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from pyparsing import Or
from sympy import Sum, degree

import tabulate
from time import perf_counter_ns
import gc

class Graph :

    def __init__(self, edges, n, m):
        self.edges = m
        self.vertex = n
        self.matrix =  np.zeros((n,n))
        self.degree = np.zeros(n)
        for edge in edges:
            u = edge[0]
            v = edge[1]
            self.matrix[u-1][v-1] = edge[2]
            self.matrix[v-1][u-1] = edge[2]
                    
        for i in range(n):
            self.degree[i-1] = sum(self.matrix[i-1])
            
    
    def contract_edge (self,u,v):
        self.degree[u] = self.degree[u] + self.degree[v] - 2 * self.matrix[u][v]
        self.degree[v] = 0
        self.matrix[u][v] = 0
        self.matrix[v][u] = 0
        points = [i for i in range(self.vertex - 1)]
        points.remove(u)
        points.remove(v)
        for w in points :
            self.matrix[u][w] = self.matrix[u][w] + self.matrix[v][w]
            self.matrix[w][u] = self.matrix[w][u] + self.matrix[w][v]
            self.matrix[v][w] = self.matrix[w][v] + 0
    


def binarySearch(alist, item):
    first = 0
    last = len(alist)-1
    found = False

    while first<=last and not found:
        pos = 0
        midpoint = (first + last)//2
        if alist[midpoint] == item:
            pos = midpoint
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1
    return (pos, found)

def random_select(c):
    r = random.choice(c) 
    pos, found = binarySearch(c, r)
    return pos + 1  


def edge_select (d, w) :
    #build comulative weights array of d
    cd = np.zeros(d.size)
    for k in range(d.size):
        cd[k] = sum(d[i] for i in range(k))
    u = random_select(cd)

    cw = np.zeros(d.size)
    for k in range(d.size):
        cw[k] = sum(w[u][i] for i in range(k))
    v = random_select(cw)

    return u, v

def contract(g, k):
    n = g.vertex
    for i in range(n - k):
        u, v = edge_select(g.degree, g.matrix)
        g.contract_edge(u,v)
    return g

def recursive_contract(g):
    n = g.vertex
    if n <= 6 :
        g = contract(g, 2)
        return g
    t = n/math.sqrt(2) + 1
    G = []
    w = []
    for i in range(2):
        G[i] = contract(g, t)
        w[i] = recursive_contract(G[i])
    return min(w)


if __name__ == '__main__':
    directory = "/Users/sofiachiarello/Desktop/unipd/advanced algorthm/Adv_algorithm/ass3/dataset"
    
    #initializing variables
    measuredTime = []
    weights=[]
    descriptions = []
    dimensions = []
    num_instances = 0 #count number of files
    num_calls = 10
    tentonine = 1000000000
    finalTotalTime = 0.0
    errors = []
    measuredTime_Size = []
    size_error = []


    #reading and sorting files
    files = os.listdir(directory)
    
    files = sorted(files)
    index = 0    # iterate over files in that directory
    for filename in files[0:1]: 
        f = os.path.join(directory, filename)
        print(f)
       
        if os.path.isfile(f): 
            file = open(f,'r')
            data = file.read().splitlines()
            n = int((data[0].split(" "))[0])
            m = int((data[0].split(" "))[1])
            data = data[1:len(data)]

            edges = []
            for i in range(m):
                point = " ".join(data[i].strip().split()) # Remove duplicate spaces in between line
                point = point.split() # Separate line by space
                edges.append([int(point[0]),int(point[1]), int(point[2])])
            #creating graph
            g = Graph(edges, n, m)

            print("n", g.vertex)
            print("m", g.edges)
            print("matrix", g.matrix)
            print("d", g.degree)

            u,v = edge_select(g.degree, g.matrix)
            g = contract(g,2)
            
            
        
        

            
            
            

            
            
             
           
            
            

    
    



  