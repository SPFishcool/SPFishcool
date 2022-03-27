import math
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

#---global optimal---
opt = []
s = open('../p07_s.txt','r')
lines = s.readlines()
opt.append(0)
for line in lines:
    opt.append(int(line))
s.close()

def calc(state):
    v_sum = 0
    w_sum = 0
    for k in range(1,count+1):
        v_sum += state[k] * item_pro[k]
        w_sum += state[k] * item_w[k]
    return (v_sum, w_sum)

def copy(a, b):
    for l in range(len(a)):
        a[l] = b[l]
#---------------initial---------------
iter_solution = []
solution = np.zeros(count+1)
best_s = np.zeros(count+1)
opt_v, opt_w = calc(opt)
best = 0
T = 200; dt = 0.95; times = 500
def init():
    init_suc = 0
    while init_suc == 0:
        for i in range(1, count+1):
            if(rand.random() < 0.5):
                solution[i] = 1
            else:
                solution[i] = 0
        v, w = calc(solution)
        if w < bag:
            init_suc = 1
            best = v
            best_s = solution
            
#---iteration function---
def iter_active():
    global best, T, solution
    #iter_list:迭代最佳解   now_lst:此次找到的新解（以迭代解開始）
    iter_lst = np.zeros(count+1)
    copy(iter_lst, solution)
    iter_value = 0
    iter_weight= 0
    #---平衡時間尋找新解---
    now_lst = np.zeros(count+1)
    copy(now_lst, iter_lst)
    iter_value, iter_weight = calc(iter_lst)
    #隨機選擇物品
    choose = rand.randint(1, count)
    if now_lst[choose] == 1:                #如果在背包內，選一物品放入
        check = 0
        while check == 0:
            change = rand.randint(1,count)
            if now_lst[change] == 0:
                now_lst[change] = 1
                check = 1
        now_lst[choose] = 0
    else:                                   #如果在背包外，隨機決定直接放入或選一物品拿出
        if rand.random() < 0.5:
            now_lst[choose] = 1
        else:
            check = 0
            while check == 0:
                change = rand.randint(1,count)
                if now_lst[change] == 1:
                    now_lst[change] = 0
                    check = 1
            now_lst[choose] = 1
            
    now_value, now_weight = calc(now_lst)
    if now_weight < bag:                    #如果沒超重，檢查是否較佳
        if now_value > iter_value:
            iter_value = now_value
            copy(iter_lst, now_lst)
        else:                               #若不較佳，則一定概率下降
            #if iter_value < opt_v:
            p = 1.0 * (now_value-iter_value)/T
            if rand.random() < math.exp(p):
                iter_value = now_value
                copy(iter_lst, now_lst)
    copy(solution, iter_lst)
    return iter_value

#---------------主程式---------------
init();
opt_a = 0
loc_v = 0
#----迭代開始----
for iteration in range(times):
    iter_cost = iter_active()
    if iter_cost == opt_v and opt_a == 0:
        print('iteration', iteration,'到達最佳解', opt_v)
        opt_a = 1
    elif iter_cost > loc_v:
        loc_v = iter_cost
        loc_iter = iteration
    T = T * dt
    iter_solution.append(iter_cost)
if opt_a == 0:
    print('iteration', loc_iter,'到達區域最佳解', loc_v)
#---------------draw---------------
plt.plot(iter_solution)
plt.title('Simulated annealing Algorithm Iteration')
plt.show()
