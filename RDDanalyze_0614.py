#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDD分析
Created on Fri Jun 11 11:02:44 2021

@author: liang-yi

2021/06/14加入中文字體、匯出圖檔
"""
### 0.載入套件
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf
import math


### 1.讀入爬蟲資料與設定存檔路徑
data = pd.read_csv("/Volumes/GoogleDrive/我的雲端硬碟/台大政研/2021程式設計/final_project/cleanedND_data/tsai_cleanND.csv")
# 設定要存圖檔的位置
save_dir = "/Volumes/GoogleDrive/我的雲端硬碟/台大政研/2021程式設計/final_project/figure"

### 2.轉換時間格式方便後續做時間處理
for i in range(len(data["Time"])):
    data["Time"].loc[i] = datetime.strptime(data["Time"].loc[i], "%Y/%m/%d")

### 3.篩選要分析的時間區間輸出為新的dataframe
date_up = datetime(2021, 4, 10) # 時間區間上界
date_down = datetime(2021, 6, 30) # 時間區間下界

df = pd.DataFrame()
for i in data["Time"].index:
    if (data["Time"].loc[i] >= date_up) & (data["Time"].loc[i] <= date_down):
        df = df.append(data.loc[i], ignore_index=False)

### 4.RDD模型設定
# 記得改各圖檔標題
## 設定斷點與時間變數
cut = datetime(2021, 5, 11) # 斷點
df["Time"] -= cut
df["Time"] = df["Time"].dt.days # 轉換成距離斷點前/後幾天

## Reaction 取對數(視情況使用)
for i in df["Reaction"].index:
    df["Reaction"].loc[i] = math.log(df["Reaction"].loc[i])

## 互動數
# 先看散佈圖
title = "蔡英文＿「宣布進入社區感染」對互動數的影響以(0511為中心)" # 圖與圖檔的標題(注意不要用\，會跟路徑符號衝突)
plt.figure(figsize=(8,8))
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta'] # 載入中文字型
ax = plt.subplot(3,1,1)
df.plot.scatter(x="Time", y="Reaction", ax=ax)
plt.title("蔡英文-「宣布進入社區感染」對互動數的影響（以5/11為中心）")

# 檢定結果
rdd_df = df.assign(threshold=(df["Time"] > 0).astype(int))
model = smf.wls("Reaction~Time*threshold", rdd_df).fit()
print(model.summary())

# RD圖
ax = df.plot.scatter(x="Time", y="Reaction", color="C0")
df.assign(predictions=model.fittedvalues).plot(x="Time", y="predictions", ax=ax, color="C1")
plt.axvline(x=0, color="b")
plt.xlabel("日期")
plt.ylabel("互動數(log)")
plt.title(title)
plt.savefig(save_dir + "/" + title + ".png", dpi=300, bbox_inches='tight')

## 留言數
# 先看散佈圖
title = "蔡英文＿「宣布進入社區感染」對留言數的影響以(0511為中心)" # 圖與圖檔的標題(注意不要用\，會跟路徑符號衝突)
plt.figure(figsize=(8,8))
ax = plt.subplot(3,1,1)
df.plot.scatter(x="Time", y="Comment", ax=ax)
plt.title("蔡英文-「宣布進入社區感染」對留言數的影響（以5/11為中心）")

# 檢定結果
rdd_df = df.assign(threshold=(df["Time"] > 0).astype(int))
model = smf.wls("Comment~Time*threshold", rdd_df).fit()
print(model.summary())

# RD圖
ax = df.plot.scatter(x="Time", y="Comment", color="C0")
df.assign(predictions=model.fittedvalues).plot(x="Time", y="predictions", ax=ax, color="C1")
plt.axvline(x=0, color="b")
plt.xlabel("日期")
plt.ylabel("留言數")
plt.title(title)
plt.savefig(save_dir + "/" + title + ".png", dpi=300, bbox_inches='tight')


## 分享數
# 先看散佈圖
title = "蔡英文＿「宣布進入社區感染」對分享數的影響以(0511為中心)" # 圖與圖檔的標題(注意不要用\，會跟路徑符號衝突)
plt.figure(figsize=(8,8))
ax = plt.subplot(3,1,1)
df.plot.scatter(x="Time", y="Share", ax=ax)
plt.title("title")

# 檢定結果
rdd_df = df.assign(threshold=(df["Time"] > 0).astype(int))
model = smf.wls("Share~Time*threshold", rdd_df).fit()
print(model.summary())

# RD圖
ax = df.plot.scatter(x="Time", y="Share", color="C0")
df.assign(predictions=model.fittedvalues).plot(x="Time", y="predictions", ax=ax, color="C1")
plt.axvline(x=0, color="b")
plt.xlabel("日期")
plt.ylabel("互動數")
plt.title(title)
plt.savefig(save_dir + "/" + title + ".png", dpi=300, bbox_inches='tight')

