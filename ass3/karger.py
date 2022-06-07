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
from sympy import Sum, degree, false
import copy
import tabulate
from time import perf_counter_ns
import gc
from threading import Thread, Event

stop_event = Event()

class Graph :

    def __init__(self, edges, n, m): #O(m)
        self.edges = m
        self.vertex = n
        self.matrix =  np.zeros((n,n))
        self.degree = np.zeros(n)
        for edge in edges:
            u = edge[0]
            v = edge[1]
            weight = edge[2]
            self.matrix[u-1][v-1] = weight
            self.matrix[v-1][u-1] = weight
            self.degree[u-1] += weight
            self.degree[v-1] += weight
                    
        #for i in range(n):
            #self.degree[i-1] = sum(self.matrix[i-1])
            
    
    def contract_edge (self,u,v): #O(n-2)
        self.degree[u] = self.degree[u] + self.degree[v] - 2 * self.matrix[u][v]
        self.degree[v] = 0
        self.matrix[u][v] = 0
        self.matrix[v][u] = 0
        points =  list(range(0,self.vertex))
        points.remove(u)
        points.remove(v)
        for w in points :
            self.matrix[u][w] = self.matrix[u][w] + self.matrix[v][w]
            self.matrix[w][u] = self.matrix[w][u] + self.matrix[w][v]
            self.matrix[v][w] = 0
            self.matrix[w][v] = 0
        
        
    

    def get_edge(self): 
        nonzeroind = np.nonzero(self.degree)[0]
        return self.degree[nonzeroind][0]



def binarySearch(a, item): #O(logn)
    first = 0
    last = len(a)-1
    found = False

    while first<=last and not found:
        pos = 0
        midpoint = (first + last)//2
        if (a[midpoint] <= item) & (a[midpoint+1] > item):
            pos = midpoint
            found = True
        else:
            if item < a[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1
    return (pos, found)

def random_select(c,k):
    r = random.randrange(0,c[k]) 

    pos, found = binarySearch(c, r) 
    if found == false & (pos == 0):
        return pos, found

    if pos == k:
        return k
    
    return pos + 1, found


def edge_select (d, w) :
    #build comulative weights array of d
    cd = [0] * g.vertex
    for k in range(d.size):
        cd[k] = cd[k-1] + d[k]
    u,f = random_select(cd,k)

    cw = [0] * g.vertex
    for k in range(d.size):
        cw[k] = cw[k-1] + w[u][k]
   
    v, f = random_select(cw,k)

    return u, v

def contract(graph, k):
    #graph = copy.deepcopy(g)
    #graph = g
    n = np.count_nonzero(graph.degree)
    for i in range(n - k): 
        u, v = edge_select(graph.degree, graph.matrix)
        graph.contract_edge(u,v)
    return graph


def recursive_contract(g):
    n = np.count_nonzero(g.degree)
    if n <= 6 :  #RIGHT
        graph = contract(g, 2)
        return graph.get_edge()

    t = math.floor((n/math.sqrt(2)) + 1)
    d = copy.deepcopy(g.degree)
    w = copy.deepcopy(g.matrix)
    g = copy.deepcopy(g)
    g1 = contract(g, t)
    m1 = recursive_contract(g1)
    g2 = contract(g, t)
    m2 = recursive_contract(g2)
    return min(m1, m2)

def plotResult(valone, valtwo, vertices):
    # sort the keys (number of vertices) of the dictionary and plot them
    plt.plot(vertices, valone, ':k')
    plt.plot(vertices, valtwo, 'r')
    plt.legend(["Measured time", "nlogn"])
    #plt.yscale("log")
    # x-axis label
    plt.xlabel('Number of Vertices')
    # frequency label
    plt.ylabel('Execution Time')
    # plot title
    plt.title('Karger Algorithm plot')
    # function to show the plot
    plt.show()

def running_code (g, num_calls):
    cut = []
    run_times = 0
    seconds_cut = []
    avg_time = 0
    tentonine = 1000000000
    for i in range(num_calls):
        g = Graph(edges, n, m)
        start_time = perf_counter_ns()
        c = recursive_contract(g)
        end_time = perf_counter_ns()
        cut.append(c)
        time = (end_time - start_time)/tentonine #seconds of the discovery cut
        seconds_cut.append((int(c), time))
        run_times += time
        if stop_event.is_set():
            break

if __name__ == '__main__':
    directory = "/Users/sofiachiarello/Desktop/unipd/advanced algorthm/Adv_algorithm/ass3/dataset"
    
    #initializing variables
    measuredTime = []
    weights=[]
    dimensions = []
    num_instances = 56
    
    total_time = 0.0
    complexity = []
    best_cut = []
    total_run_times = []

    #reading and sorting files
    files = os.listdir(directory)
    
    files = sorted(files)
    index = 0    # iterate over files in that directory
    for filename in files[40:45]: 
        f = os.path.join(directory, filename)

       
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
            ##----APPEND DIMENSIONS OF GRAPH-----
            dimensions.append(n) 
            num_calls = int(math.log(n)**2)
           
            cut = []
            #calculate times for one graph
            gc.disable()
            run_times = 0
            seconds_cut = []
            avg_time = 0
            for i in range(num_calls):
                g = Graph(edges, n, m)
                start_time = perf_counter_ns()
                c = recursive_contract(g)
                end_time = perf_counter_ns()
                cut.append(c)
                time = (end_time - start_time)/tentonine #seconds of the discovery cut
                seconds_cut.append((int(c), time))
                run_times += time
                if stop_event.is_set():
                    break

            gc.enable()

            
 
           
            
            total_run_times.append(run_times)
            avg_time = ((run_times)/num_calls)
            ###----APPEND MEASURED TIME OF ALG-----
            measuredTime.append(avg_time)
            ##----APPEND WEIGHTS OF THE GRAPH-----
            cut = int(min(cut))
            #print(cut)
            weights.append(cut)

            total_time += avg_time
            ##----APPEND THEORICAL COMPLEXITY-----
            comp = (n**2)*(math.log(n)**3)
            complexity.append(comp)
            ##----APPEND DISCOVERY TIME-----
            seconds_cut = sorted(seconds_cut, key=lambda item: item[0])
            
            best_cut.append(seconds_cut[0][1])

            #MANCA
            # - STAMPARE GRAFICI
            
    zipFileSizeSol = zip(files[40:45], dimensions, weights, measuredTime)


  
    tableRunOutput = tabulate.tabulate(zipFileSizeSol, headers=['File',  'N', 'Solution' ,'AvgTime'], tablefmt='orgtbl')
    print(tableRunOutput)
    print("Total time for all alg/num_calls",total_time)
    

    zipFileBestCut = zip(files[40:45], dimensions,weights, best_cut, total_run_times)
    
    tableOutput = tabulate.tabulate(zipFileBestCut, headers=['File', 'N', 'Minimun Cost Cut', 'Discovery Time','TotalTime'], tablefmt='orgtbl')
    print(tableOutput)

    print("Weights",weights)
    print("measured time", measuredTime)
    print("complexity", complexity)
    print("discovery time", best_cut)
    print("Totoal time", total_run_times)
    
    #plotResult(measuredTime, complexity, dimensions)


            
        
        

            
            
            

            
            
             
           
            
            

    
    



  