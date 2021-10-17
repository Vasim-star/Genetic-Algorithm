# import all the important packages like matplotlib,cv2,pygame,time,sys,random and numpy
from sys import maxsize
from itertools import permutations
from typing import Counter
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor,Button
import cv2
import matplotlib.image as img
import numpy as np
import random
import pygame
import sys
import math
import time
import os.path


value=[]
x_value=[]
count=[]
save_path=[]
#calculating the distance between the points slected from the image windows 




def distancenew(x,y):
    final=list()
    n=len(x)
    for i in range(n):
        for j in range(n):
            A1=[x[i],y[i]]
            A1=np.array(A1)
            A2=[x[j],y[j]]
            A2=np.array(A2)
            v=A2-A1
            b=np.abs(np.linalg.norm(A1-A2))
            final.append(b*209/120.49554802956189)
    final=np.array(final).reshape(n,n)
    return final
temp=[6,7,8,9,12]
for i in temp:
    text="{}_points.png".format(i)
    print(text,'\n')
    img = cv2.imread("{}".format(text))
    # img = cv2.imread("dronepath.png")

    fig,ax=plt.subplots()

    # p,=plt.plot(img)

    p=plt.imshow(img)
    cursor=Cursor(ax,horizOn=True, vertOn=True,color='green',linewidth=2)
    global x,y
    x=[] 
    y=[] 
    def oneclick(event):
        x1,y1=event.xdata,event.ydata
        x.append(x1)
        y.append(y1)



    fig.canvas.mpl_connect('button_press_event',oneclick)
    plt.show()



    adj=distancenew(x,y)
    # print(adj)
    N=len(x)
    pygame.init()


    # implementation of traveling Salesman Problem
    def travellingSalesmanProblem(graph, s):

        # store all vertex apart from source vertex
        vertex = []
        for i in range(N):
            if i != s:
                vertex.append(i)
        path=[0]
        # store minimum weight Hamiltonian Cycle
        min_path = maxsize
        min_path_vertex=vertex 
        next_permutation=permutations(vertex)
        for i in list(next_permutation):
            # store current Path weight(cost)
            current_pathweight = 0

            # compute current path weight
            k = s
            for j in i:
                current_pathweight += graph[k][j]
                k = j
            current_pathweight += graph[k][s]
            # update minimum
            if min_path<current_pathweight:
                min_path=min(min_path, current_pathweight)
                # path_append=np.array(i)
            elif min_path>current_pathweight:
                min_path=min(min_path, current_pathweight)
                path_append=np.array(i)
        
        # path_append=np.asarray(path_append)
        path=np.append(path,path_append)
        return min_path,path


    # Driver Code
    # matrix representation of graph
    graph = adj
    s = 0
    start=time.time()
    final=travellingSalesmanProblem(graph, s)
    end=time.time()
    print('Code 1:Minimum distance travelled:',final[0])
    print('Code 1:Path obtained:',final[1])
    print('Code 1:The number of points selected are :',N)
    print('Code 1:Time involved :',end-start)

    totalNum = N # Total number of destinations 
    # popNum = [100]
    popNum = [2000]#,20,100,200,1000,2000,10000,20000]
    font = pygame.font.Font('freesansbold.ttf', 15)
    WIDTH = 600
    HEIGHT = 600
    PERCENTAGE = 0.5 # How much of the current population to crossover for the next generation

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Travelling Salesman Problem")

    for v in range(len(popNum)):
        # print('running',v)
        class City:
            def __init__(self, x, y, i):
                self.x = x
                self.y = y
                self.num = i
                self.text = font.render(str(self.num), False, (255, 255, 255))

            def display(self):
                pygame.draw.circle(screen, (255, 255, 255), (self.x, self.y), 5)

        #Initializing the coordinates of the cities imported from the image set as mentioned before.
        cities = [City(x[i],y[i],i) for i in range(N)] 
        class Route:
            def __init__(self):
                self.distance = 0
                self.cityPath = random.sample(list(range(N)), totalNum)

            def display(self):
                for i, cityNum in enumerate(self.cityPath):
                    pygame.draw.line(screen, (0, 0, 255), (cities[self.cityPath[i]].x, cities[self.cityPath[i]].y), \
                                    (cities[self.cityPath[i-1]].x, cities[self.cityPath[i-1]].y))

            def calcDistance(self):
                distance = 0
                for i, cityNum in enumerate(self.cityPath):
                    distance += (math.sqrt((cities[self.cityPath[i]].x - cities[self.cityPath[i-1]].x)**2 + \
                                        (cities[self.cityPath[i]].y - cities[self.cityPath[i-1]].y)**2))*209/120.49554802956189
                    
                self.distance = distance
                return distance


        population_ori = [Route() for i in range(popNum[v])]
        # print(popNum[v],'Those value were used')
        population=population_ori

        #Sorts the population ie, the distance of the route
        def sortPop():
            global population
            population.sort(key = lambda x : x.distance, reverse = False)
            return
        '''
        Takes the top PERCENTAGE of the population for a particular generation and 
        produces a new population replacing the non essential members with new ones 
        '''
        def crossover():
            global population
            updatedPop = []
            updatedPop.extend(population[: int(popNum[v]*PERCENTAGE)])

            for i in range(popNum[v]- len(updatedPop)):
                index1 = random.randint(0, len(updatedPop) - 1)
                index2 = random.randint(0, len(updatedPop) - 1)
                while index1 == index2:
                    index2 = random.randint(0, len(updatedPop) - 1)
                parent1 = updatedPop[index1]
                parent2 = updatedPop[index2]
                p = random.randint(0, totalNum - 1)
                child = Route()
                child.cityPath = parent1.cityPath[:p]
                notInChild = [x for x in parent2.cityPath if not x in child.cityPath]
                child.cityPath.extend(notInChild)
                updatedPop.append(child)
            population = updatedPop
            return

        # bg = pygame.image.load("6_points.png")

        # The image size is same as the pygame windows

        display_width = 600 
        display_height = 600

        gameDisplay = pygame.display.set_mode((display_width,display_height))

        running = True
        counter = 0
        i=0
        best = random.choice(population)
        minDistance = best.calcDistance()
        clock = pygame.time.Clock()
        start=time.time()
        while True:
            # gameDisplay.blit(bg, (0, 0))
            best.display()
            if counter >= popNum[v]-1:
                end=time.time()
                break
            clock.tick(60)
            pygame.display.update()
            screen.fill((0, 0, 0))
            for city in cities:
                city.display()
                screen.blit(city.text, (city.x - 20, city.y - 20))
            for element in population:
                element.calcDistance()

            sortPop()
            crossover()
            
            for element in population:
                if element.distance < minDistance:
                    minDistance = element.calcDistance()
                    #value.append(minDistance)
                    best=element

                elif element.distance == minDistance:
                    counter += 1

            for element in population:
                if element.distance < minDistance:
                    minDistance = element.calcDistance()
                    #value.append(minDistance)
                    best=element

                elif element.distance == minDistance:
                    counter += 1
            # print(minDistance)
            value.append(best.calcDistance())
            x_value.append(i+1)


        print("Code 2:Minimum distance travelled: {}".format(minDistance))
        print("Code 2:Path obtained : {}".format(best.cityPath))
        print("Code 2:The number of points selected are :",N)
        print("Code 2:Time Involved:",end-start)
        text='Number of Crossover population is {}'.format(popNum[v])
        filename1='Fitness_count {}.jpg'.format(popNum[v])
        filename='Number of Population {}.jpg'.format(popNum[v])
        # plt.plot(value,label=text)
        # plt.ylabel('Fitness Function')
        # plt.legend()
        # plt.title(text)
        # plt.savefig(filename1)
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # best.display()
        # pygame.display.update()
        # pygame.image.save(screen,filename)
        # time.sleep(10)
            


