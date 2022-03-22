import numpy as np
import pandas as pd

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

#---------------Dynamic Programming---------------
table = np.zeros((count+1,bag+1))
#---table[i][j] = max{table[i-1][j], table[i-1][j-weight[i]] + cost[i]}---
for i in range(1,count+1):
    for j in range(min_weight,bag+1):
        not_contain = table[i-1][j]
        if j < item_w[i]:
            table[i][j] = not_contain
        else:
            contain = table[i-1][j-item_w[i]] + item_pro[i]
            if contain > not_contain:
                table[i][j] = contain
            else:
                table[i][j] = not_contain
        '''
        contain = table[i-1][j-item_w[i]] + item_pro[i]
        not_contain = table[i-1][j]
        if contain > not_contain:
            table[i][j] = contain
        else:
            table[i][j] = not_contain
         '''
#---output the table(Dynamic Programming Table)---
table_df = pd.DataFrame(table)
table_df.to_csv('output.csv')

#---check item in the bag---
inbag = []
remain = bag
for n in range(count,0,-1):
    if table[n][remain] > table[n-1][remain]:
        inbag.append(n)
        remain = remain - item_w[n]


output = []
for i in range(len(inbag),0,-1):
    output.append(inbag[i-1])

print(output)
