import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#------------------------使用訓練集訓練------------------------
train = pd.read_csv("train.csv")
#------------特徵標準化------------
scaler = StandardScaler()
scaler.fit(train.drop('price_range',axis=1))
scaled_features_train = scaler.transform(train.drop('price_range',axis=1))
train_feat = pd.DataFrame(scaled_features_train, columns=train.columns[:-1])

x = train_feat
y = train['price_range']
#---分割資料成訓練資料和測試資料---
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05,random_state=0)

print('size of train set =',len(y_train))
print('size of test set =', len(y_test))
print('class 0 in train set:',np.sum(y_train==0))
print('class 1 in train set:',np.sum(y_train==1))
print('class 2 in train set:',np.sum(y_train==2))
print('class 3 in train set:',np.sum(y_train==3))
#----根據準確率找尋最適合的K值----
max_rate = 0
for k in range(10,1000):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    hit_rate = knn.score(x_test,y_test)
    if max_rate < hit_rate:
        max_rate = hit_rate
        k_fit = k
        pred_train = pred_i

print('訓練集訓練最佳K值=',k_fit,'，score=',max_rate)

#---------製作混淆矩陣---------
matrix = confusion_matrix(y_test, pred_train, labels=[0,1,2,3])
print('混淆矩陣如下：')
print(matrix)

#------------模型評估------------
col_sum = np.zeros(4)
row_sum = np.zeros(4)
precision = np.zeros(4)
recall = np.zeros(4)
f1_score = np.zeros(4)

for i in range(4):
    col_sum[i] = matrix[0][i] + matrix[1][i] + matrix[2][i] + matrix[3][i]
    row_sum[i] = matrix[i][0] + matrix[i][1] + matrix[i][2] + matrix[i][3]
total = np.sum(col_sum)

accuracy = (matrix[0][0]+matrix[1][1]+matrix[2][2]+matrix[3][3])/total

for i in range(4):
    precision[i] = matrix[i][i] / col_sum[i]
    recall[i] = matrix[i][i] / row_sum[i]
    f1_score[i] = (2*recall[i]*precision[i])/(recall[i]+precision[i])
    print('class',i,'Precision:',precision[i])
    print('class',i,'Recall:',recall[i])
    print('class',i,'F1 Score:',f1_score[i])
    
print('The model\'s Accuracy:',accuracy)

#------開始分類test.csv------
test = pd.read_csv("test.csv")

scaler.fit(test.drop('id',axis=1))
scaled_features_test = scaler.transform(test.drop('id',axis=1))
test_feat = pd.DataFrame(scaled_features_test, columns=test.columns[1:])

X_train = train_feat
Y_train = train['price_range']
X_test = test_feat

knn = KNeighborsClassifier(n_neighbors=k_fit)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
test['price_range'] = pred

#------將分類結果匯出------
test_range = test.loc[:,['id','price_range']]
test_range.to_csv('output.csv',index=False)
print('test.csv已分類完成，結果匯入於output.csv')

