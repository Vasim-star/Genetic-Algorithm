# import all the important packages like matplotlib,cv2,pygame,time,sys,random and numpy

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

def finalxy(path,xin,yin):
    x=list()
    y=list()
    xin=np.array(xin)
    yin=np.array(yin)
    for i in range(len(path)):
        x.append(xin[path[i]])
        y.append(yin[path[i]])
    return x,y


def bat_opti(path,slope,bat_points,x,y,speed):
    vas=path_plot(path,x,y)
    print(distancenew(vas[0],vas[1]))
    dis=distancenew(vas[0],vas[1])[0][:1]


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

def path_plot(path,x,y):
    n=len(path)
    print(path)
    path=np.append(path,path[0])
    print(path)
    path=np.array(path)
    d=[]
    for i in range(n):
        s=np.sqrt(np.power(x[path[i+1]]-x[path[i]],2)+np.power(y[path[i+1]]-y[path[i]],2))*209/120.49554802956189
        d.append(s)
    return d

img =cv2.imread("dronepath.png")

fig,ax=plt.subplots()
# p,=plt.plot(img)

p=plt.imshow(img)
cursor=Cursor(ax,horizOn=True, vertOn=True,color='green',linewidth=2)
global x,y
x_old=[] 
y_old=[] 
def oneclick(event):
    x1,y1=event.xdata,event.ydata
    x_old.append(x1)
    y_old.append(y1)
    print(x_old,y_old)

fig.canvas.mpl_connect('button_press_event',oneclick)
plt.show()
N_sta=int(input("Number of ground station selected at the end"))
L=len(x_old)-N_sta
x=x_old[:L]
y=y_old[:L]

    






# print(x,y)


adj=distancenew(x,y)
print(adj)
N=len(x)
pygame.init()


totalNum = N # Total number of destinations 
# popNum = [100]
popNum = [10000]#,20,100,200,1000,2000,10000,20000]
font = pygame.font.Font('freesansbold.ttf', 15)
WIDTH = 600
HEIGHT = 600
PERCENTAGE = 0.5 # How much of the current population to crossover for the next generation

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Genetic Algorithm")

for v in range(len(popNum)):
    print('running',v)
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
    print(popNum[v],'Those value were used')
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

    bg = pygame.image.load("dronepath.png")
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

        gameDisplay.blit(bg, (0, 0))
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
        print(minDistance)
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
    filename='route.png'
    pygame.image.save(screen,filename)
    # time.sleep(10)
    Watt_rating=129.96 #Whr rating in Drone battery.
    Current=21.30673581444 
    Voltage=21.76263293986667
    Velocity=4#m/s
    print(Current,Voltage)
    Time_Max=Watt_rating*60/(Current*Voltage)
    print(Time_Max,'minutes')

    A=distancenew(x_old,y_old)
    B=A[L:,:L]
    print(np.shape(B))
    Time_inspection=2#minuites
    Energy=np.divide(path_plot(best.cityPath,x,y),(Velocity*60))+Time_inspection
    B=np.divide(B,(Velocity*60))+Time_inspection
    Total_Energy=Time_Max*0.5
    E_charge=Time_Max
    E_left_path=list()
    p=best.cityPath
    p.append(p[0])
    #p=[0,1,2,3,4,5]
    max_path=max(p)
    a=max_path
    j=[]
    i=0
    while i<N_sta:
        a=a+1
        j.append(a)
        i=i+1
        print(i)

    p_new=list()
    p_new.append(p[0])
    for i in range(len(p)-1):
    #     p_new.append(p[i])
        if Total_Energy>Energy[p[i]]:
            Total_Energy=Total_Energy-Energy[p[i]]        
        elif Total_Energy<Energy[p[i]]:
            min1=B[0,p[i]]
            v_selected=0
            for v in range(1,len(j)):
                if(B[v,p[i]]<min1):
                    min1=B[v,p[i]]
                    v_selected=v
    #         print(min1,v_selected,i)
            p_new.append(j[v_selected])
            Total_Energy=E_charge
            Total_Energy=Total_Energy-B[v_selected,p[i]]-B[v_selected,p[i+1]]
        p_new.append(p[i+1]) 
        # print(np.add(p_new,1),Total_Energy)

    print(p,'Before the battery')
    print(p_new,'Before the battery')
    xfinal,yfinal=finalxy(p_new,x_old,y_old) 
    plt.imshow(img)
    plt.plot(xfinal,yfinal,'red')
    plt.savefig('final_with_battery.png')
    plt.show(block=False)
    plt.pause(30)
    plt.close()





