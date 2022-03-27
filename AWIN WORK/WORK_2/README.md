CNN卷積神經網路


※程式流程：
1.將資料夾內圖片依照種類以dictionary型態存取
2.每種類圖片皆以7:3比例分成train set, test set
3.重洗圖片順序
4.圖片數值正規化（÷255)
5.建立模型
  （Conv2D→Conv2D→MaxPooling→dropout→Conv2D→Conv2D→MaxPooling→dropout→
    flatten→Dense→dropout→Dense）
6.訓練並測試
7.輸出隨機十筆資料比對


※程式構想：
    為了讓圖片比例能夠平等，決定先以dictionary一類型存放圖片再shuffle→split→shuffle
嘗試過多種模型組合，目前以兩次的「卷積→卷積→最大池化」效果最佳，並且經過多次測試20次
訓練（echos=20）已是目前最佳效益。


※模型評估：
    模型精確度為0.71，雖作為分類模型尚遠遠不及，但至少能做出基本分類，或許是處理圖片處
理的不夠好導致失真誤導預測，以下為取樣10筆結果。


※取樣結果：
1396526833_fb867165be_n.jpg 	預測為 daisy 正確解為 daisy
5990626258_697f007308_n.jpg 	預測為 tulip 正確解為 rose
3502251824_3be758edc6_m.jpg 	預測為 tulip 正確解為 tulip
15987457_49dc11bf4b.jpg 	預測為 daisy 正確解為 dandelion
3011223301_09b4e3edb7.jpg 	預測為 tulip 正確解為 tulip
15452909878_0c4941f729_m.jpg 	預測為 tulip 正確解為 tulip
5718510972_9a439aa30a_n.jpg 	預測為 tulip 正確解為 tulip
494803274_f84f21d53a.jpg 	預測為 rose  正確解為 rose
5670916806_df4316006f_n.jpg 	預測為 daisy 正確解為 tulip
8008258043_5457dd254b_n.jpg 	預測為 tulip 正確解為 daisy