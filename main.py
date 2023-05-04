# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:07:07 2023

@author: Mohamed Sagou
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class Data:
    def __init__(self,n_entrees=100, types_de_proprite=["apartment", "house", "villa"], locations=["Paris", "Lyon", "Marseille"], start_date = datetime(2010, 1, 1),
    end_date = datetime(2022, 12, 31)):
        self.n_entrees = n_entrees
        self.types_de_proprite = types_de_proprite
        self.locations = locations
        self.location_index = {"Paris":1,
                               "Lyon":2,
                               "Marseille":3}
        self.start_date = start_date
        self.end_date =end_date
        self.data = ""
    
    
    
    def date_contruction_vente(self):
        L_construction = []
        L_vente = []
        
        for i in range(self.n_entrees):
            rand1 =np.random.randint((self.end_date - self.start_date).days)
            rand2 = np.random.randint((self.end_date - self.start_date).days)
            if rand1 > rand2:
                date_construction = self.start_date + timedelta(days=rand2)
                date_vente = self.start_date + timedelta(days=rand1)
                L_construction.append((date_construction))
                L_vente.append(date_vente)
            else:
                date_vente = self.start_date + timedelta(days=rand2)
                date_construction = self.start_date + timedelta(days=rand1)
                L_construction.append((date_construction))
                L_vente.append(date_vente)
                
        return [L_construction, L_vente]
    
    
    def export_data(self):
        self.data.to_csv('data.txt', sep='\t', index=False)
        
    def import_data(self):
        self.data = pd.read_csv('data.txt', sep='\t')
    
    
        
    def create_data(self):
        dates = self.date_contruction_vente()
        self.data = pd.DataFrame({
            "ID": range(1, self.n_entrees+1),
            "Property Type": np.random.choice(self.types_de_proprite,self.n_entrees),
            "Nombre de chambres": np.random.randint(1, 5, self.n_entrees),
            "Nombre de salles de bain": np.random.randint(1, 3, self.n_entrees),
            "Prix": np.random.randint(100000, 1000000, self.n_entrees),
            "Emplacement": np.random.choice(self.locations, self.n_entrees),
            "Date de construction": dates[0],
            "Date de vente": dates[1]
        })
        return self.data


    # Nettoyez et préparez les données pour l'analyse
    
    def sup_val_mq(self):
        self.data.dropna(inplace=True)
        return
    
    def date_converter(self):
        self.data["Date de construction"] = pd.to_datetime(self.data["Date de construction"])
        self.data["Date de vente"] = pd.to_datetime(self.data["Date de vente"])
        return
    
    def sup_valeurs_aberrantes_extremes(self):
        self.data = self.data[(np.abs(self.data["Prix"] - self["Prix"].mean()) / self["Prix"].std()) < 3]
        return
    
    
    
    # Réalisez une analyse exploratoire des données
    
    def analyse_exploratoire(self):
        self.data.describe()
        return
    
    # un model pour une regression linaire
    # Split the data into training and testing sets
    def reg_lin(self):
        X = self.data[["Emplacement", "Nombre de salles de bain"]]
        y = self.data["Prix"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    
        lr = LinearRegression()
        lr.fit(X_train, y_train)
    
        y_pred = lr.predict(X_test)
    
        r2 = r2_score(y_test, y_pred)
        print("R-squared: {.2}".format(r2))
'''
data = Data()
data.create_data()
data.export_data()
data.import_data()
data.analyse_exploratoire()
print(data.data)
'''


