# 特徵工程 D22 - D32

## 簡介
* 監督式學習：
  前處理-->探索式數據分析-->特徵工程-->模型選擇
* 特徵工程是事實對應到後續評估分數的轉換
* 由於資料包含類別型(⽂文字型)特徵以及數值型特徵，所以最⼩小的特徵⼯工程⾄至少要包含⼀一種類別編碼(範例例使⽤用標籤編碼)，以及⼀一種特徵縮放⽅方法(範例例使⽤用最⼩小最⼤大化)

![特徵工程](https://pic2.zhimg.com/80/20e4522e6104ad71fc543cc21f402b36_720w.jpg)
[來源：知乎-特征工程到底是什么？](https://www.zhihu.com/question/29316149)

## 數值型特徵-去除偏態
* [偏態：Skewness, 峰度：Kurtosis](https://blog.csdn.net/u013555719/article/details/78530879)
* 什麼情況需要去除偏態？
  * 離群資料比例太高
  * 平均值沒有代表性
* 去除偏態方法
  * 對數去偏（log1p,加一取log)（適用大於等於零的資料）
  * 方根去偏 (數值減去最小值開根號，最大值有限適用)
  * 分布去偏 (boxcox, $\lambda$ 介於0到1.5)

## 類別型特徵 - 基礎處理
* 基礎編碼
  * 標籤編碼(Label Encoding)：類似流水號，缺點是分數大小順序沒有意義
  * 讀熱編碼(One Hot Encoding)：將不同類別分立獨立一欄，缺點是需要較大空間與計算時間
* 非深度學習模型主要為樹狀，建議採用標籤編碼；反之則不易收斂
* 當特徵重要性⾼高，且可能值較少時，才應該考慮獨熱編碼
> 補充：[数据预处理：独热编码（One-Hot Encoding）和 LabelEncoder标签编码](https://www.twblogs.net/a/5baab6e32b7177781a0e6859/zh-cn/)

## 類別型特徵 - 均值編碼
* 均值編碼(Mean Encoding):使用目標值的平均值取代原本的類別型特徵（如果類別特徵看起來與目標值顯著相關
* 平滑化(Smoothing):依照紀錄筆數在總平均與類別平均間取折衷
> 補充：[平均数编码：针对高基数定性特征（类别特征）的数据预处理/特征工程](https://zhuanlan.zhihu.com/p/26308272)



## 範例1:房價預測精簡版 流程：
* 讀取訓練與測試資料，注意data_path
* 進行最小的特徵工程，載入標籤編碼與最小最大化套件
```python
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
```
* 存取ids（預測輸出需要，用以識別每個預測值）
```python
ids = df_test['Id']
```
* 將train, test data做合併，因為不論何種特徵工程, 都需要對 train / test 做同樣處理
```python
df = pd.concat([df_train,df_test])
```
* LabelEncoder 與 MinMaxScaler
```python
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
```
>補充: [label encoding](https://medium.com/@PatHuang/初學python手記-3-資料前處理-label-encoding-one-hot-encoding-85c983d63f87)、[標準化與正規化方法](https://aifreeblog.herokuapp.com/posts/54/data_science_203/)

* 文字型/類別型欄位, 先補缺 'None' 後, 再做標籤編碼；數值補缺 -1
```python
for c in df.columns:
    if df[c].dtype == 'object': # 如果是文字型 / 類別型欄位, 就先補缺 'None' 後, 再做標籤編碼
        df[c] = df[c].fillna('None')
        df[c] = LEncoder.fit_transform(df[c])
    else: # 其他狀況(本例其他都是數值), 就補缺 -1
        df[c] = df[c].fillna(-1)
    # 最後, 將標籤編碼與數值欄位一起最大最小化, 因為需要是一維陣列, 所以這邊切出來後用 reshape 降維
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
```
>補充：[fit(), transform(), fit_transform()差別](https://blog.csdn.net/weixin_38278334/article/details/82971752)

## 範例2 房價預測 - 降低資料偏態
* 過濾出數值型欄位
```python
#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')
# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
```
> 補充：[zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表](https://www.runoob.com/python/python-func-zip.html)
* 作圖 （直觀看資料分佈）
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df['LotArea'][:train_num])
plt.show()
```
* 計算交叉驗證分數
```python
# 計算基礎分數
df_mm = MMEncoder.fit_transform(df)
train_X = df_mm[:train_num]
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
```
> 補充： [交叉驗證避免過度依賴一種資料切分方式，分成training, testing data](https://ithelp.ithome.com.tw/articles/10197461

## 範例3 房價預測 - 觀查標籤編碼與獨編碼熱的影響
* 標籤編碼
```
df_temp = pd.DataFrame()
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
```
* 計算時間
```
start = time.time()
time.time() - start
```
* 獨熱編碼
```
df_temp = pd.get_dummies(df)
```
