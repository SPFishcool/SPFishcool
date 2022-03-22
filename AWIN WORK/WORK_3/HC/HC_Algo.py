import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt

#---------------load data from .txt---------------
#---capacity---
capacity = open('../p07_c.txt','r')
bag = int(capacity.readline())
capacity.close()

#---profit of item---
profits = open('../p07_p.txt','r')
item_pro = []
count = 0
lines = profits.readlines()
item_pro.append(0)
for line in lines:
    item_pro.append(int(line))
    count = count + 1
profits.close()

#---weight of item---
min_weight = bag
weight = open('../p07_w.txt','r')
item_w = []
lines = weight.readlines()
item_w.append(0)
for line in lines:
    item_w.append(int(line))
    if(min_weight > int(line)):
        min_weight = int(line)
weight.close()

#---------------initial---------------
solution = np.zeros(count+1)
remain = bag
order = list(range(1,count+1))
unchoose = []

#---shuffle the order,put them in the bag until it is full---
rand.shuffle(order)
ischoose = []
iteration = []
cost = 0
for init in order:
    if remain > item_w[init]:
        remain = remain - item_w[init]
        solution[init] = 1
        ischoose.append(init)
        cost = cost + item_pro[init]
    else:
        unchoose.append(init)
iteration.append(cost)

#---------------start iteration---------------
for i in range(0,500):
    change = rand.choice(unchoose)  #change:要放入背包的物品（取代）
    changed= rand.choice(ischoose)  #changed:要拿出背包的物品（被取代）
    differ = item_w[change]-item_w[changed]
    cost_differ = item_pro[change]-item_pro[changed]
    if cost_differ > 0 and remain > differ:
        unchoose.append(changed)    
        unchoose.remove(change)
        ischoose.append(change)
        ischoose.remove(changed)
        solution[changed] = 0
        solution[change] = 1
        remain = remain - differ
        cost = cost + cost_differ
        print('Iteration',i+1,'times find new solution:',cost)
    iteration.append(cost)

#---------------draw---------------
print(solution)
plt.plot(iteration)
plt.title('Hill Climbing Algorithm Iteration')
plt.show()
    
    
    
    
