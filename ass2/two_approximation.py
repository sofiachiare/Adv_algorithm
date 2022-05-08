#from extended_int import Infinite
import math
import heapq
from pickle import NONE
import time
import os
from pathlib import Path
import numpy as np
import tabulate


def prims(graph, start):
    key = dict() 
    minHeap = []
    parent = dict()
    #This set adds all the nodes of the Minimum Spanning Tree
    mst = [start]
    #initialize all the parents to null and the key to infinity for each node in the graph
    for v in graph:
        parent[v] = np.NaN
        key[v] = math.inf

    #The starting node has a key of 0 and it is first pushed to the heap   
    key[start] = 0
    heapq.heappush(minHeap,(0,start))
    
    #check if the minHeap is not empty, if it's empty then no more vertices to visit
    while minHeap:
        #pop the vertex with minimum weight from minHeap
        weight, u = heapq.heappop(minHeap)
        #The popped node has a minimum weight value, then we add it to the mst
        if u not in mst:
            #add the node to mst 
            mst.append(u)

        for v, weight in graph[u]:
            #Only update the parent, key, and minHeap if the edge is minimum and the node is not already a part of MST
            if v not in mst and v in key and weight < key[v]:  
                parent[v] = u
                key[v] = weight
                #add the node with the weight to the minHeap
                heapq.heappush(minHeap,(weight,v))

      
    #calculates the minimum weight obtained by Prim's Algorithm
    totalWeight = 0 
    for v in key: #all the stored key at the end are equal to the minimum weight between v and parent[v]
        totalWeight = totalWeight + key[v] 
    return mst, totalWeight


def converting(x):
    PI = 3.141592;
    deg = int(x); 
    min = x - deg; 
    rad = PI * (deg + 5.0 * min/ 3.0) / 180.0; 

    return rad


def claculateGEODist(x,y,x2,y2):
    RRR = 6378.388;

    q1 = math.cos( converting(y) - converting(y2) );
    q2 = math.cos( converting(x) - converting(x2) );
    q3 = math.cos( converting(x) + converting(x2) );
    return int( RRR * math.acos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0)

def createGraph(nodes, dimension, edge_type):
    graph = {}
    for node,x,y in nodes:
        edge = list(map(int, [node,x,y]))
        if edge[0] not in graph: 
            graph[edge[0]] = []
        for node2, x2,y2 in nodes:
            if node == node2:  break
            edge2 = list(map(int, [node2,x2,y2]))
            #add the node to the graph and initialize it with an empty list 
            if edge2[0] not in graph: 
                graph[edge2[0]] = []
            if weight_type == "EUC_2D":
                distance = round(math.dist([x,y],[x2,y2]))
            else: distance = claculateGEODist(x,y,x2,y2)

            #For each node u, append the map of second node v along with the weight (u,v) of the edge
            #For each node v, append the map of second node u along with the weight (u,v) of the edge
            graph[edge[0]].append([edge2[0],distance])
            graph[edge2[0]].append([edge[0], distance])

    return graph   

if __name__ == '__main__':
    directory = "/Users/sofiachiarello/Desktop/unipd/advanced algorthm/Adv_algorithm/ass2/tsp_dataset"
    
    #initializing variables
    measuredTime = []
    weights=[]
    descriptions = []
    dimensions = []
    num_instances = 0 #count number of files
    num_calls = 1 #call the prim function this specific number of time.
    finalTotalTime = 0.0
    errors = []
    optimalsolution = [7542,3323,6528,35002, 18659688 , 426, 40160, 134602, 21282, 21294, 50778, 6859, 7013]

    print('Weight:\t\t\tTime:\t\t\tVertices:\t\t\tFile:')

    #reading and sorting files
    files = os.listdir(directory)
    files = sorted(files)
    index = 0    # iterate over files in that directory
    for filename in files: 
        print(filename)
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f): 
            
            file = open(f,'r')
            data = file.read().splitlines()

            problem_name = (data[0].split(' ')[-1])
            descriptions.append(data[2][9:])
            dimension = int((data[3].split(' ')[-1]))
            dimensions.append(dimension)
            weight_type = (data[4].split(' ')[-1])
            if weight_type == "EUC_2D":
                data = data[6:len(data)-1]
            else: 
                weight_format =  (data[5].split(' '))[1] 
                display_data_type =  (data[6].split(' '[-1]))
                data = data[8:len(data)-1]


            nodes = []
            for i in range(dimension-1):
                point = " ".join(data[i].strip().split()) # Remove duplicate spaces in between line
                point = point.split() # Separate line by space
                nodes.append([int(point[0]),float(point[1]), float(point[2])])

            #print(nodes, " size: ", dimension)
            graph = createGraph(nodes, dimension, weight_type)
           
            #print(graph)
            #calculate the time of prim's algorithm on one graph
            start = time.time()
            #for i in range(num_calls):
            starting_node = next(iter(graph))
            mst, weight = prims(graph,starting_node)
            mst.append(starting_node)

            weight += graph[starting_node][mst[len(mst)-2]][1]

            end_start = float(time.time() - start) / num_calls 
            finalTotalTime = finalTotalTime + end_start

            weights.append(weight)
            measuredTime.append(float(end_start)) 
            errors.append(float((weight-optimalsolution[index])/(optimalsolution[index]*1.00)))
            index = index +1
            print(f"{str(weight):20s} {str(end_start):30s} {str(dimension):20s} {str(Path(f).stem):10s}")

    print("------------------------------------------------\n")
    print("weights: ", weights, " len: ", len(weights))
    print("optimal sol: ", optimalsolution, " len: ", len(optimalsolution))
    
    #for i in range(len(weights)):
     #   errors = str(float((weights[i]-optimalsolution[i])/(optimalsolution[i]*1.00)))

    #printing table size | run_times | m*n | C | ratio
    zipFileSizeSol = zip(files, descriptions, dimensions, weights, optimalsolution, measuredTime, errors)
  
    tableRunOutput = tabulate.tabulate(zipFileSizeSol, headers=['File', 'Description', 'N', 'Solution', 'Optimal Solution','Time', 'Error'], tablefmt='orgtbl')
    print(tableRunOutput)

    #for i in range(len(weights)):
    #    print(f"{str(files[i]):20s} {str(descriptions[i]):25s} {' ':10s}{(str(weights[i])):15s}{(str(measuredTime[i])):25s}{(str(errors[i])):30s}")
    #print(100*"-")

    print("Total time: (s) ", finalTotalTime)

    



