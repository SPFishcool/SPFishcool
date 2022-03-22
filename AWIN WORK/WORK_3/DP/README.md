0-1 Knapsack Problem By Dynamic Programming

※程式流程:
1.建Dynamic Programming Table（column:背包重量限制、row:物品編號），並建立外內迴圈i、j
2.在選擇物品與不選擇物品中取最大值
2-1.選擇物品i，則table[i][j] = table[i-1][j-weight[i]] + cost[i]
2-2.不選擇物品i，則table[i][j] = table[i-1][j]
3.重複step.2直到i、j達到最大值
4.輸出table為output.csv檔方便Debug
5.照id大到小檢查在背包內的物品，如果背包裡有物品，則table[i][j]必定不等於（大於）table[i-1][j]
  ，再依照剩下物品之重量依序檢查
6.輸出選取之物品ID


※程式構想：
    Dymanic Programing就是將subproblem之solution存至陣列來預防重複計算，再建立遞迴式來完成此演
算法，在迴圈方面我從物品中最小重量開始來減少執行時間，再利用程式流程step 5的規則找出背包內的物
品。