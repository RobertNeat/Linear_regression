#dane używane w projekcie - zbiór BostonHousing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#zadanie 2.1 - wyznaczenie korelacji i generowanie wykresow korelacji między cechami niezależnymi a zależną (medianową ceną mieszkania)

data_excel = pd.read_excel("practice_lab_2.xlsx")
nazwy_kolumn = list(data_excel.columns)
wartosci_kolumn = data_excel.values
tablica = np.array(wartosci_kolumn)

korelacja = data_excel.corr()


for a in range(len(nazwy_kolumn)):
        plt.xlabel(nazwy_kolumn[a])
        plt.scatter(wartosci_kolumn[:,a],wartosci_kolumn[:,13])
        plt.show()




#zadanie 2.2 - wielokrotne przetestowanie regresji liniowej
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
bh_data = pd.read_excel('practice_lab_2.xlsx')
bh_arr = bh_data.values
X, y = bh_arr[:,:-1], bh_arr[:,-1]

def wielokrotne_testowanie(powtorzenia):
    for a in range(powtorzenia):
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, shuffle=False)
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        mape = mean_absolute_percentage_error (y_test, y_pred)
        print("powtorzenie:",a,"=",mape)

wielokrotne_testowanie(5)



#zadanie 2.3 - wielokrotne przetestowanie regresji liniowej z suwaniem wartości odstających
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
bh_data = pd.read_excel('practice_lab_2.xlsx')
bh_arr = bh_data.values
X, y = bh_arr[:,:-1], bh_arr[:,-1]



def wielokrotne_testowanie(powtorzenia):
    for a in range(powtorzenia):
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, shuffle=False)
        #usuniecie danych ostających
        outliers = np.abs((y_train-y_train.mean())/y_train.std())>3
        x_train_no_outliers = X_train[~outliers,:]
        y_train_no_outliers = y_train[~outliers]
        y_train_mean = y_train.copy()
        y_train_mean = y_train.mean()

        linReg = LinearRegression()
        linReg.fit(x_train_no_outliers, y_train_no_outliers)
        y_pred = linReg.predict(X_test)
        mape = mean_absolute_percentage_error (y_test, y_pred)
        print("powtorzenie:",a,"=",mape)

wielokrotne_testowanie(5)