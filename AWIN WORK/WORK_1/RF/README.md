Random Forest模型

※程式流程:
1.匯入train.csv進行訓練，分割比例為0.05
2.利用準確度選取C值（懲罰係數）
3.用最適合之C值製作混淆矩陣並做模型評估
4.相同模型開始分類test.csv
5.將id與對應price_range輸出至output.csv


※程式構想：
    因為Random Forest就是數個Decision Tree組成，因此與Decision Tree我多次建模型找出Score最高之模型。

※模型評估：
    在分類訓練時此訓練及各類參考數較平均，「支持度」差距不大。
    「準確度」平均約為0.87，比Decision Tree高，但因為有多棵Decision Tree，選出來的模組最高Score最
高約0.92，雖比Decision Tree時最高0.97低，但相對其範圍之「精確度」、「召回度」與「F1分數」也比較平
衡表現可以說是比Decision Tree穩定