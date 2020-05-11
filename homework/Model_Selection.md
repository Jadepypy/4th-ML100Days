# Model Selection D33 - D46

## D34 - 訓練/測試集切分的概念

##  D35 - Regression vs. Classification
* 機器學習的監督式學習主要分為回歸問題(目標值為實數)與分類問題（目標值為類別）
* Regression vs. Classification
  * output type: 連續數字 vs. 離散
  * 目的：線 vs. decision boundary
  * 衡量方式： RSS vs. accuracy vs.
* 回歸問題可轉換為分類問題
* 分類問題
  * 二元分類 vs. 多元分類
  * Multi-class vs. Multi-Label

## D36 - Evaluation Metrics
* 設定各項指標來評估模型的正確性
  * 最常見為準確率（Accuracy）：正確分類樣本數/總樣本數
* 評估指標 - 回歸: Prediction 與 Ground Truth 差距
  * MAE, Mean Absolute Error, [0, $\infty$]  
  * MSE, Mean Square Error, [0, $\infty$]
  * R-square, [0, 1]
* 評估指標 - 分類: 觀察實際值與預測值的正確程度
  * AUC, Area Under Curve, [0, 1]: 衡量曲線下的面積，可考量所有閾值下的準確性 （二分類）
  * F1 - Score(Precision, Recall), [0, 1] ： 觀察特定類別
    * Precision: 模型判定瑕疵，樣本確實為瑕疵的比例例 True/False
    * Recall: 模型判定的瑕疵，佔樣本所有瑕疵的比例例 Positive/Negative
    * F1 - Score兩者調和平均數
  * 混淆矩陣(Confusion Matrix):縱軸為模型預測、橫軸為正確答案
  * top-k accuracy: k 代表模型預測前 k 個類別有包含正確類別即為正確
> 補充：[ROC curves and Area Under the Curve explained](https://www.dataschool.io/roc-curves-and-auc-explained/)整理: ROC curve
