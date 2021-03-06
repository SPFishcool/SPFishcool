SVM模型

※程式流程:
1.匯入train.csv進行訓練，分割比例為0.05
2.利用準確度選取C值（懲罰係數）
3.用最適合之C值製作混淆矩陣並做模型評估
4.相同模型開始分類test.csv
5.將id與對應price_range輸出至output.csv


※程式構想：
    在撰寫程式時發現kernel為線性效果最好，因此將kernel設定為'linear'，並找尋適當的懲罰係數C
經迴圈測試C=1時最佳。

※模型評估：
    在分類訓練時此訓練及各類參考數較平均，「支持度」差距不大。
    「準確度」為四種模型中最高0.99，可得知此模型最適合分類本資料集，可能是因為SVM會在多條分界
線找出一條寬度最大的最佳解，會很接近實際分界線。
    「精確度」除「範圍2」為0.96其餘皆為1.0，也可能是因為測試資料不夠多看不出太多判斷錯誤的部
分，但依照此訓練集來看模型表現還是很好的。
    「召回度」除「範圍3」為0.97其餘皆為1.0，在混淆矩陣中，只有一筆「範圍2」資料被判定為「範圍3」
    「F1分數」：綜合精確度與召回度，F1分數也很高，也可見到「範圍2」與「範圍3」之判斷可能會又些許
錯誤。
    