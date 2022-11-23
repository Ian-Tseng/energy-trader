# Energy trader




### Environment
```
The following is the steps to build pipenv environment:
$ cd energy_trader
$ pipenv install 
$ pipenv shell
$ python --3.8

```
### Usage

```
$ python main.py --generation --consumption --output
    
```




### Introduction


### 



```


### Model
```
## 資料前處理

作業中的 training_data 是將每戶 2018/01/01 - 2018/08/31 用電的資訊以 csv 檔案格式個別存放，時間單位是 hour 。因為作業的問題是希望透過過去七天的訊息，來產生下一天的預測結果。因此在建構訓練數據的流程是將時間序列的資料以每 7+1 天的範圍來擷取，前七天當作 feature ，第八天當作 target ，示意圖如下。時間單位則是將一天 24 小時分成 4 時段，讓時間的變數可以數值化，當作訓練時的特徵使用。

<img src="./val-Process.drawio.png"/> 

對於數據採用 min-max Scale 作法將數據限制在 [-1, 1] 區間。
本次作業使用 3 種特徵數據，編號以及對應描述如下所示。

- time_f1: min-max scale after (hour / 6)
- generation_f1: min-max scale generation column
- consumption_f1: min-max scale consumption column


## 驗證資料

訓練模型的時候，我需要從中切出一部分當作假的測試資料，用於評估是否有辦法抓到模式。訓練資料總共有 50 戶的用電資料，所以從中切出 20% 的用戶當作驗證資料。

<img src='./val-Validation.drawio.png' />

## 模型架構

模型的輸入有 time_f1, generation_f1, consumption_f1 三種特徵，輸出也是，差別在於輸入是七天的時間的特徵，輸出是一天的時間的預設數值。

<img src='./val-model.drawio.png' />

##  實驗結果

使用 Adam 優化器進行訓練，訓練時使用的 loss function 皆採用 MSE 來進行優化。對於損失的計算，對 generation_f1 以及 consumption_f1 增加 50% 的損失權重。初步可以看到訓練集的損失曲線有往下的趨勢，意味過程中有抓到資料的 pattern ，另外從驗證集上也可以看到下降的趨勢，所以在驗證集上也是有抓到相同的 pattern 。

<img src='./Figure_1.png' />

##  模型輸出

固定上述使用的參數，將所有的訓練資料用於模型的訓練，並儲存成 .pt 檔案。

##  用於測試資料

路徑都不要改，使用 `python test.py` 指令可以獲得測試資料下一天的預測結果。

```


### Agent


```
## Preprocess
1. 計算 供需比 (request- supply rate)
    1) Idea: 價格根據供需比在變動。 供需比的公式如下： 需求量/ 供應量。 如果供需比顯示較高的值表示需求量大，供應量少， 價格趨向高， 相反的， 需求量少， 供應量大， 則呈現較小的值， 價格趨向低。
    Reference url: https://www.youtube.com/watch?v=PHe0bXAIuk0

    2) 我們試著找出在這交易資訊裡的供需比， 公式如下： sum(generate of all_familys)/ sum(consumption of all familys)

    3) 以下為顯示不同時段、不同星期下的供需比的範例。 
    

    | Weekday | Hour     | Request-Storage rate |  
    |---------|----------|----------------------|
    | 5       | 00:00:00 | 835.68               |   
    | 5       | 01:00:00 | 333.27               | 
    | 5       | 02:00:00 | 118.87               | 

2. 計算不同時間、 不同星期的平均交易量與平均成交價格
    1) 我們計算在不同時段、 不同星期的平均交易量與平均成交價格
    2) 以下為顯示不同時段、 不同星期的平均交易量與平均成交價格範例。 以上兩筆與下兩筆為對照， 可以看到在特定時段、特定星期的交易量與交易價個成反比，交易量越多成交價格越低， 相反的交易量越少， 成交越高。 

    | Weekday | Hour     | Mean trade volume | Mean trade price |   |
    |---------|----------|-------------------|------------------|---|
    | 1       | 15:00:00 | 0.192             | 2.241            |   |
    | 2       | 16:00:00 | 0.186             | 2.257            |   |
    | 6       | 20:00:00 | 16.318            | 1.668            |   |
    | 4       | 22:00:00 | 20.932            | 1.763            |   |

    3) 下圖顯示交易量與成交價格的關係。
    <img src='./trade_volume_price_relation.png' />


2. 預測出target日期的產出與消耗
    1) 先從Model 預測出target日期的產出與消耗。
    3) 計算每個時段產出的總和與消耗的總和。 
    
3. 價格策略
    1) 參考價格使用目標時段，目標星期的平均價個The reference price using the trader price in target time.
    2) The price we take from range 0.9 to 1.2 as discount on reference price, the step= 0.05. 
    3) If the trader price is invalid, which means the target trader at that time is not success and defualt refernce price would use market price. The dscount range from 0.5 to 0.8.
 

4. Amount strategy
    1) We use sum of generate and sum of consumption as reference amount.
    2) We try to buy sum of the consumption in a lowest RSR rate at target time.
    3) We try to sell all of the remaining of energy in a highest RSR rate at target time. The remaining of energy = (sum of generation+ sum of buy)- sum of consumption. The amount would multiply by 0.8  in order to prevent the excession of sum of consumption
    
5. Action strategy
    1) We initiate sell action if RSR rate is higher than a threshold at target time. In constrast, take buy action if RSR rate is higher than a threshold at target time.
    2) We get the threshold by find a value lower than percentitle 20 as threshold for buy action and high than percentile 70.
    


6. Improve profit by bid result
    1) As 1. 4) mentioned, we get RSR from trade information data. The trade information data show when the needed request and sell request is higher or lower, we  based on this information to decide when is a good time to initate acion.

## Action trigger condition
```
1. We initate buy action if RSR rate is lower than a threshold and target volume = trade history volume* 0.8  at target time.
2. We initate sell action if RSR rate is higher than a threshold at targe time and target volume = trade history volume at target time.
```




