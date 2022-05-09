#from asyncio.windows_events import INFINITE, NULL
import math
import heapq
from pickle import NONE
import time
import os
from tkinter import W
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import tabulate


def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)


def prims(graph, start):
    key = dict() 
    minHeap = []
    parent = dict()
    result = []
    #This set adds all the nodes of the Minimum Spanning Tree
    mst = [start]
    #initialize all the parents to null and the key to infinity for each node in the graph
    for v in graph:
        parent[v] = np.NaN
        key[v] = math.inf

    #The starting node has a key of 0 and it is first pushed to the heap   
    key[start] = 0
    heapq.heappush(minHeap,(0,start, start))
    
    #check if the minHeap is not empty, if it's empty then no more vertices to visit
    while minHeap:
        #pop the vertex with minimum weight from minHeap
        weight, p,u = heapq.heappop(minHeap)
    
        #The popped node has a minimum weight value, then we add it to the mst
        if u not in mst:
            #add the node to mst 
            mst.append(u)
            result.append((weight, p,u))

        for v, weight in graph[u]:
            #Only update the parent, key, and minHeap if the edge is minimum and the node is not already a part of MST
            if v not in mst and v in key and weight < key[v]:  
                parent[v] = u
                key[v] = weight
                #add the node with the weight to the minHeap
                heapq.heappush(minHeap,(weight,u,v))

    #calculates the minimum weight obtained by Prim's Algorithm
    totalWeight = 0 
    for v in key: #all the stored key at the end are equal to the minimum weight between v and parent[v]
        totalWeight = totalWeight + key[v] 
        
    return totalWeight, result


def converting(x):
    PI = 3.141592
    deg = int(math.floor(x))
    min = x - deg
    return PI * (deg + 5.0 * min/ 3.0) / 180.0


def calculateGEODist(x,y,x2,y2):
    RRR = 6378.388;

    q1 = math.cos( converting(y) - converting(y2) );
    q2 = math.cos( converting(x) - converting(x2) );
    q3 = math.cos( converting(x) + converting(x2) );
    return int( RRR * math.acos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0)

def createGraph(nodes, dimension, weighttype):
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
            if 'EUC_2D' in weighttype:
                distance = round(math.dist([x,y],[x2,y2]))
            else: distance = calculateGEODist(x,y,x2,y2)

            #For each node u, append the map of second node v along with the weight (u,v) of the edge
            #For each node v, append the map of second node u along with the weight (u,v) of the edge
            graph[edge[0]].append((edge2[0], distance))
            graph[edge2[0]].append((edge[0], distance))

    return graph  

def preorder(graph,vertex,path):
    path += [vertex]
    for neighbor in graph[vertex]:
        if neighbor not in path: 
            path = preorder(graph,  neighbor,path)
        
    return path

def return_distance(graph, mst):
    weight = 0
    for i in range(len(mst)-1):
        array = graph[mst[i]]
        for j,w in array:
            if j == mst[i+1]:
                weight += w
    return weight


def createTreeFromEdges(edges):
    tree = {}
    for weight,v1, v2 in edges:
        tree.setdefault(v1, []).append(v2)
        tree.setdefault(v2, []).append(v1)
    return tree

def plotResult(xy_graph, yname, legend):
    # sort the keys (number of vertices) of the dictionary and plot them
    plt.plot( *zip(*sorted(xy_graph) ), ':k')
    print(yname, " " , sorted(xy_graph))
    plt.legend([legend])
    #plt.yscale("log")
    # x-axis label
    plt.xlabel('Number of Vertices')
    # frequency label
    plt.ylabel(yname)
    # plot title
    plt.title('Two Approximation Algorithm plot')
    # function to show the plot
    plt.show()

if __name__ == '__main__':
    directory = "/Users/sofiachiarello/Desktop/unipd/advanced algorthm/Adv_algorithm/ass2/tsp_dataset"
    
    #initializing variables
    measuredTime = []
    weights=[]
    dimensions = []
    num_instances = 0 #count number of files
    num_calls = 1 #call the prim function this specific number of time.
    finalTotalTime = 0.0
    errors = []
    optimalsolution = [7542,3323,6528,35002, 18659688 , 426, 40160, 134602, 21282, 21294, 50778, 6859, 7013]
    measuredTime_Size = []
    size_error = []
    #reading and sorting files
    files = os.listdir(directory)
    files = sorted(files)
    index = 0    # iterate over files in that directory
    for filename in files: 
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f): 
            
            file = open(f,'r')
            data = file.read().splitlines()
            problem_name = (data[0].split(' ')[-1])
            dimension = int((data[3].split(' ')[-1]))
            dimensions.append(dimension)
            weight_type = (data[4].split(' ')[-1])
            #print(weight_type)
            if 'EUC_2D' in weight_type:
                data = data[6:len(data)]
            else: 
                weight_format =  (data[5].split(' '))[1] 
                display_data_type =  (data[6].split(' '[-1]))
                data = data[7:len(data)]
                if data[0] == "NODE_COORD_SECTION":
                    data = data[1:len(data)]


            nodes = []
            for i in range(dimension):
                if 'EOF' in data[i]: break 
                point = " ".join(data[i].strip().split()) # Remove duplicate spaces in between line
                point = point.split() # Separate line by space
                nodes.append([int(point[0]),float(point[1]), float(point[2])])

            graph = createGraph(nodes, dimension, weight_type)
           
            #calculate the time of 2-approximation algorithm on one graph
            start = time.time()
            #for i in range(num_calls):
            starting_node = next(iter(graph))
            weight,edges = prims(graph,starting_node)

            dfrsTree = createTreeFromEdges(edges)

            visited = set() # Set to keep track of visited nodes of graph.
            path = []
            preorderTree = preorder(dfrsTree , starting_node,path)         

            preorderTree.append(starting_node)

            finalWeight = return_distance(graph, preorderTree)


            end_start = float(time.time() - start) / num_calls 
            finalTotalTime = finalTotalTime + end_start
            error = float((finalWeight-optimalsolution[index])/(optimalsolution[index]*1.00))
            weights.append(finalWeight)
            measuredTime.append(float(end_start)) 
            errors.append(error)
            measuredTime_Size.append((int(dimension),float(end_start)))
            size_error.append((int(dimension),float(error)))
            index = index +1
            
    

    #printing table:
    zipFileSizeSol = zip(files, dimensions, weights, optimalsolution, measuredTime, errors)
  
    tableRunOutput = tabulate.tabulate(zipFileSizeSol, headers=['File',  'N', 'Solution', 'Optimal Solution','Time', 'Error'], tablefmt='orgtbl')
    print(tableRunOutput)

    print("Total time: (s) ", finalTotalTime)

    plotResult(size_error, "Error", "Errors")
    plotResult(measuredTime_Size, 'Execution Time', "Execution Time")




