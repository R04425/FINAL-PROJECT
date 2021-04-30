# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import json
import warnings
warnings.filterwarnings("ignore")


def country_list(data):
    data.rename(columns={'Country Name':'country_name'},inplace=True)
    data['country_name'] = data['country_name'].apply(lambda row: row.lower())
    lists = data['country_name'].unique().tolist()
    with open('country_list.json','w', encoding='utf-8') as f:
        json.dump(lists, f, ensure_ascii=False,indent=4)
    return lists, data

def select_country(data,country):
    data = data.loc[data['country_name']==country]
    data.drop(['country_name','Country Code','Indicator Name','Indicator Code'],axis=1,inplace=True)
    data = data.T
    data.dropna(inplace=True)
    data = data.reset_index()
    return data

def prediction_model(data):
    x = data.iloc[:, 0].values.reshape(-1,1)
    y = data.iloc[:, 1].values.reshape(-1,1)
    model = LinearRegression().fit(x,y)
    return model

def prediction(model, year):
    return int(model.coef_[0][0] * year + model.intercept_[0])


def main():
    country = input("Please input the country name: ").lower()
    year = int(input("Please input the year to predict: "))
    data = pd.read_csv('pop.csv')
    lists, data = country_list(data)
    if country in lists:
        data = select_country(data, country)
        model = prediction_model(data)
        result = prediction(model,year)
        print(f"\n Result: {country.upper()} population in {year} will be {result:,d}")
    else:
        print('kindly check country name spelling from country_list.json')
    
if __name__ == "__main__":
    main()


# next step
#pickle the model
# criar uma flaskapp spyder

#1 app.py 
#2. helper
# dictionary
#3. main.html
