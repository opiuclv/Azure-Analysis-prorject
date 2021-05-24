# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:45:37 2018

@author: Chiu-Cheng-Chung
"""


#導入模塊
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import datetime

def dateRange(start, end, step=1, format="%Y-%m-%d"):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days, step)]

#搜尋日期範圍
date = dateRange("2016-12-30", "2017-01-01")

for d in range(len(date)):
    
    #搜尋日期
    day = date[d]
    
    #搜尋URL
    index = ('http://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=467490&stname=%25E8%2587%25BA%25E4%25B8%25AD&datepicker='+day)
    
    #請求資料
    res = requests.get(index) #REQUEST
    
    #資料剖析
    soup = BeautifulSoup(res.text, "lxml")
    pattern = soup.find_all("td")
    
    #定義Title
    title = np.array(["觀測時間","測站氣壓","海平面氣壓","氣溫","露點溫度","相對溼度","風速","風向",
                      "最大陣風","最大陣風風向","降水量","降水時數","日照時數","全天空日射量","能見度"])
    #創建表格
    frame = np.ones([24,15])
    
    row = 0 #計算小時用的基底
    column = 0 #決定是屬於哪個Title
    start = 4

    for item in pattern:
        if start > 0:
            start = start - 1
            continue
        
        if column == 15:
            column = 0
            row = row + 1
            if row == 24:
                break
        try:
            frame[row, column] = float(item.string)
            
        except:
            frame[row, column] = np.nan
        column = column + 1
            
        

    #合併Title和表格
    new_frame = np.vstack((title, frame))
    
    #儲存至excel
    new_frame = pd.DataFrame(new_frame)
    writer = pd.ExcelWriter(day + '.xlsx')
    new_frame.to_excel(writer, sheet_name = 'Data', index = False, header = False)
    writer.save()




