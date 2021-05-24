#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:31:27 2019

@author: opiuclv
"""

# 使用迴歸分析來預測台新金股票(2887)
import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()

def get_BaseFiles() : #2017 2018年份
    # 指定要列出所有檔案的目錄
    mypath = r"/Users/opiuclv/Desktop/Taishin2887Base"
    
    # 取得所有檔案與子目錄名稱
    Basefiles = listdir(mypath)  #遍歷整份檔案
    Basefiles.pop(5) #在run的時候會跑出一個DS.Store隱藏暫存檔 要把它踢掉
    Basefiles.sort(key=lambda x:int(x[14:19])) #排序檔案名稱
    
    for i in range(len(Basefiles)): # 使用loop集合資料
        #content = content.append(files[i])
        if i == 0:
            content = pd.read_csv("/Users/opiuclv/Desktop/Taishin2887Base/" + Basefiles[i]).iloc[1:,:]
        else:
            content = np.vstack((content, pd.read_csv("/Users/opiuclv/Desktop/Taishin2887Base/" + Basefiles[i]).iloc[1:,:]))
        

    content = pd.DataFrame(content) #將集合好的資料存成csv方便分析
    content.to_csv("TaishinBasefile.csv",index = False)
    return Basefiles

def get_file(): #2019年份
    # 指定要列出所有檔案的目錄
    mypath = r"/Users/opiuclv/Desktop/Taishin2887"
    
    # 取得所有檔案與子目錄名稱
    files = listdir(mypath)  #遍歷整份檔案
    files.pop(2) #在run的時候會跑出一個DS.Store隱藏暫存檔 要把它踢掉
    files.sort(key=lambda x:int(x[13:15])) #排序檔案名稱
    
    for i in range(len(files)): # 使用loop集合資料
        #content = content.append(files[i])
        if i == 0:
            content = pd.read_csv("/Users/opiuclv/Desktop/Taishin2887/" + files[i]).iloc[1:,:]
        else:
            content = np.vstack((content, pd.read_csv("/Users/opiuclv/Desktop/Taishin2887/" + files[i]).iloc[1:,:]))
        

    content = pd.DataFrame(content) #將集合好的資料存成csv方便分析
    content.to_csv("Taishin.csv",index = False)
    return files


files = get_file()
Basefiles = get_BaseFiles()

priceBase = pd.read_csv("/Users/opiuclv/Desktop/TaishinBasefile.csv").iloc[:,6] #讀檔並設定參數
price = pd.read_csv("/Users/opiuclv/Desktop/Taishin.csv").iloc[:,6] #讀檔並設定參數
#true_dateBase = pd.read_csv("/Users/opiuclv/Desktop/TaishinBasefile.csv").iloc[:,0] #讀檔並設定參數
#true_date = pd.read_csv("/Users/opiuclv/Desktop/Taishin.csv").iloc[:,0] #讀檔並設定參數
#plot_date = np.hstack((true_dateBase, true_date))

dateBase = []
date = []

for i in range(len(priceBase)) :
    dateBase.append(i + 1)

for i in range(len(price)) :
    date.append(len(dateBase) + i + 1)


"""
date = list(map(int, date)) #字串陣列轉數字
price = list(map(int, price))
"""

dateBaseInt = list(map(int, dateBase)) #字串陣列轉數字
priceBaseInt = list(map(int, priceBase))


# Creates a linear regression from the data points
m,b = np.polyfit(dateBaseInt, priceBaseInt, 1)

# This is a simple y = mx + b line function
def f(x):
    return m*x + b


# Pick the Linear Regression model and instantiate it
model = LinearRegression(fit_intercept=True)


dateBase=np.array(dateBase) #轉成array而不是list不然無法fit
priceBase=np.array(priceBase)
dateBaseInt=np.array(dateBaseInt)
#date=np.array(date) #轉成array而不是list不然無法fit
#price=np.array(price)

# Fit/build the model

model.fit(dateBase[:, np.newaxis], priceBase)
mean_predicted = model.predict(dateBaseInt[:, np.newaxis])


plt.scatter(dateBase, priceBase) 
plt.scatter(date, price) 
plt.plot(f(np.hstack((dateBase, date))))
#plt.plot(dateBase, mean_predicted)
plt.title('Scatter plot of Taishin(2887) Stock price vs date')
plt.xlabel('2017,2018(1-472)   2019(473-617)', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.xticks(rotation=120) #x軸項目顯示斜的
#plt.xlim(0, 25) #顯示幾筆數目
#plt.yticks(rotation=120) 

# Prints text to the screen showing the computed values of m and b
#print(' y = {0} * x + {1}'.format(m, b))
sns.regplot(dateBase, priceBase)
plt.show()

print(' y = {0} * x + {1}'.format(model.coef_[0], model.intercept_))

