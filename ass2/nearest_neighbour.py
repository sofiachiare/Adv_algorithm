import math
import heapq
from pickle import NONE
import time
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import tabulate
from time import perf_counter_ns
import gc

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
            else: 
                
                distance = claculateGEODist(x,y,x2,y2)
              

            #For each node u, append the map of second node v along with the weight (u,v) of the edge
            #For each node v, append the map of second node u along with the weight (u,v) of the edge
            graph[edge[0]].append([edge2[0],distance])
            graph[edge2[0]].append([edge[0], distance])

    return graph  

def return_distance(graph, u, v):
    for k, weight in graph.get(u):
        if k == v:
            return weight
    return 0
    


def nearest_neighbour_algorithm(graph):
    points = [0] * len(graph)
    for i in range(len(points)):
        points[i] = i+1
    
    #first node of the cicle
    current = points[0]
    #Hamiltonian cicle
    tour = [current]
    #points to visit
    points.remove(current)

    while len(points) > 0:
        next = points[0]
        for point in points:
            #find the next node that minimize w(current, point)
            if return_distance(graph, current, point) < return_distance(graph, current, next):
                next = point
            
        tour.append(next)
        points.remove(next)
        current = next
    tour.append(tour[0])
    #computing cost of the path
    cost = 0
    for i in range(len(tour)-1):
        cost += return_distance(graph, tour[i], tour[i+1])

    cost += return_distance(graph, tour[i+1], tour[0])
    return tour, cost

def plotResult(xy_graph, yname, legend):
    # sort the keys (number of vertices) of the dictionary and plot them
    plt.plot( *zip(*sorted(xy_graph) ), ':k')
    #print(yname, " " , sorted(xy_graph))
    plt.legend([legend])
    #plt.yscale("log")
    # x-axis label
    plt.xlabel('Number of Vertices')
    # frequency label
    plt.ylabel(yname)
    # plot title
    plt.title('Nearest Neighbour Algorithm plot')
    # function to show the plot
    plt.show()

if __name__ == '__main__':
    directory = "/Users/sofiachiarello/Desktop/unipd/advanced algorthm/Adv_algorithm/ass2/tsp_dataset"
    
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
            descriptions.append(data[2][9:])
            dimension = int((data[3].split(' ')[-1]))
            dimensions.append(dimension)
            weight_type = (data[4].split(' ')[-1])
            if weight_type == "EUC_2D":
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
            
             #calculate the time 
            start_time = perf_counter_ns()
            for i in range(num_calls):
                tour, cost = nearest_neighbour_algorithm(graph)
            end_time = perf_counter_ns()
            gc.enable()
           
            
            print(index)
            end_start = ((end_time - start_time)/num_calls)/tentonine 
            finalTotalTime = finalTotalTime + end_start
            weights.append(cost)
            measuredTime.append(float(end_start)) 
            error = float((cost-optimalsolution[index])/(optimalsolution[index]*1.00))
            errors.append(error)
            measuredTime_Size.append((int(dimension),float(end_start)))
            size_error.append((int(dimension),float(error)))
            index = index +1

    
    zipFileSizeSol = zip(files, dimensions, weights, optimalsolution, measuredTime, errors)


  
    tableRunOutput = tabulate.tabulate(zipFileSizeSol, headers=['File',  'N', 'Solution', 'Optimal Solution','Time', 'Error'], tablefmt='orgtbl')
    print(tableRunOutput)

    print("Total time: (s) ", finalTotalTime)
    print(measuredTime_Size)

    plotResult(size_error, "Error", "Errors")
    plotResult(measuredTime_Size, 'Execution Time', "Measured Time")
  


