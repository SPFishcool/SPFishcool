import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import tree

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

#----根據準確率找尋最適合的max_depth----
max_rate = 0
score_lst = []
for i in range(1000):
    tree_clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=20)
    tree_clf = tree_clf.fit(x_train,y_train)
    score = tree_clf.score(x_test, y_test)
    pred_i = tree_clf.predict(x_test)
    score_lst.append(score)
    if max_rate < score:
        max_rate = score
        i_fit = i+1
        pred_train = pred_i
        fit_tree = tree_clf

print('訓練集訓練最佳score=',max_rate)
print('score平均:',np.mean(score_lst))

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

fit_tree.fit(X_train,Y_train)
pred = fit_tree.predict(X_test)
test['price_range'] = pred

#------將分類結果匯出------
test_range = test.loc[:,['id','price_range']]
test_range.to_csv('output.csv',index=False)
print('test.csv已分類完成，結果匯入於output.csv')
