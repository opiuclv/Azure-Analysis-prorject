# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 22:00:30 2018

@author: User
"""

# mac使用excel開啟UTF-8 csv檔 100%會遇到 亂碼！！！！！！ 所以要轉文字格式

import datetime
import requests
import json
import csv

def dateRange(start, end, step=1, format="%Y-%m-%d"):  #設定datarange函式
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days, step)]


# ***********************************************************************************************************************************

#搜尋日期範圍
date = dateRange("2017-11-01", "2017-12-31") # 設定時間範圍

testnum =  date[0]+date[1]+date[2]+date[3]+date[5]+'.'+date[8]+date[9] # 先給testnum初始 **因為不要"-"符號

for d in range(len(date)): #用迴圈跑日期

    #搜尋日期值
    day = date[d][0]+date[d][1]+date[d][2]+date[d][3]+date[d][5]+date[d][6]+date[d][8]+date[d][9] # 給day初始值 **因為不要"-"符號值
    
    if testnum[5] != day[5] :  # 一個月只要抓一次

        url_twse = 'http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date='+day+'&stockNo=2887&_=1538746599700'

        res = requests.get(url_twse) # get網頁內容
        s = json.loads(res.text) # 把json格式轉給python
    
        #outputfile = open(r'台灣證券市場(台新金)'+day[5]+'月'+'.csv', 'w', newline='' ) 
        outputfile = open(r'Taishin(2887)'+ '201'+ day[3] + day[4] + day[5] +'.csv', 'w', newline='' ) 
        outputwriter = csv.writer(outputfile)
        outputwriter.writerow(s['title'])
        outputwriter.writerow(s['fields'])
        for data in (s['data']):
            outputwriter.writerow(data)
    
        outputfile.close()
        
    testnum =  date[d][0]+date[d][1]+date[d][2]+date[d][3]+date[d][5]+date[d][6]+date[d][8]+date[d][9]

        
        
        
        
        
        
        
        
        
