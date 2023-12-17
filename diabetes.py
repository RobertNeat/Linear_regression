#Robert Sereda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import danych
from sklearn.datasets import load_diabetes
data = load_diabetes()
dane = pd.DataFrame(data.data, columns=data.feature_names)


nazwy_kolumn = list(dane.columns)
wartosci_kolumn = dane.values
tablica = np.array(wartosci_kolumn)

# Zbadanie korelacji między zbiorami danych
tablica= dane.corr()
fig, ax=plt.subplots(len(nazwy_kolumn), len(nazwy_kolumn), figsize=(40,40))
for a in range(len(nazwy_kolumn)):
    for b in range(len(nazwy_kolumn)):
        ax[a,b].scatter(dane.iloc[:,a],dane.iloc[:,b])
fig.tight_layout()


#Największe korelacje dotyczą danych o indeksach:
# 4 i 5 - (s1-total_serum_choresterol i s2-low-density_lipoproteins)
# 2 i 6 - (bmi i high_density_lipoproteins)



#podział zbioru danych na testowe i trenująca, regresja liniowa
def wielokrotne_testowanie(powtorzenia):
    from sklearn.linear_model import LinearRegression #dołączenie bibliotek używanych w funkcji
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error
    X, y = tablica[:,:-1], tablica[:,-1]

    s=0
    for a in range(powtorzenia):
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, shuffle=True) #podział na część testową i uczącą
        linReg = LinearRegression() #wyznaczenie regresji liniowej
        linReg.fit(X_train, y_train) #dopasowanie regresji do wartości uczących
        y_pred = linReg.predict(X_test) #przetestowanie nauczonego modelu regresji
        mape = mean_absolute_percentage_error (y_test, y_pred) #wyznaczenie średniego procentu błędu
        s+=mape
    return s/powtorzenia

print ("Średni błąd bez wykluczania wartości odstających:",wielokrotne_testowanie(5))
#Średni błąd bez wykluczania wartości odstających: 2.259758824021351


#generowanie wykresu pudełkowego
def generate_boxplot(tablica):
    X, y = tablica[:,:-1], tablica[:,-1]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
    plt.boxplot(y_train)
    plt.title("Wykres pudełkowy dla tablicy \"tablica\"")

generate_boxplot(tablica)

#podział zbioru danych na testowe i trenująca, regresja liniowa, usunięcie wartości odstających)
def wielokrotne_testowanie_no_outliers(powtorzenia):
    from sklearn.linear_model import LinearRegression #dołączenie bibliotek używanych w funkcji
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error
    X, y = tablica[:,:-1], tablica[:,-1]

    s=0
    for a in range(powtorzenia):
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, shuffle=True) #podział na część testową i uczącą
        outliers = np.abs((y_train-y_train.mean())/y_train.std())>2 #wyznaczenie części odstającej
        X_train_no_outliers = X_train[~outliers,:] #stworzenie cech niezależnych z wykluczeniem wierszy z wartościami odstającymi
        y_train_no_outliers = y_train[~outliers] #stworzenie cechy niezależnej z wykluczeniem wierszy z wartościami odstającymi
        linReg = LinearRegression() #wyznaczenie regresji liniowej
        linReg.fit(X_train_no_outliers, y_train_no_outliers) #dopasowanie regresji do wartości uczących
        y_pred = linReg.predict(X_test) #przetestowanie nauczonego modelu regresji
        mape = mean_absolute_percentage_error (y_test, y_pred) #wyznaczenie średniego procentu błędu
        s+=mape
    return s/powtorzenia

print ("Średni błąd z wykluczeniem wartości odstających:",wielokrotne_testowanie_no_outliers(5))
#Średni błąd z wykluczeniem wartości odstających: 1.7027269379390524

def wielokrotne_testowanie_no_outliers_zastapienie_cech(powtorzenia):
    from sklearn.linear_model import LinearRegression #dołączenie bibliotek używanych w funkcji
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error
    X, y = tablica[:,:-1], tablica[:,-1]

    s=0
    for a in range(powtorzenia):
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, shuffle=True) #podział na część testową i uczącą
        outliers = np.abs((y_train-y_train.mean())/y_train.std())>2 #wyznaczenie części odstającej

        #zastąpienie wartości odstających
        y_train_mean = y_train.copy()
        y_train_mean[outliers] = y_train.mean()

        linReg = LinearRegression() #wyznaczenie regresji liniowej
        linReg.fit(X_train, y_train_mean) #dopasowanie regresji do wartości uczących
        y_pred = linReg.predict(X_test) #przetestowanie nauczonego modelu regresji
        mape = mean_absolute_percentage_error (y_test, y_pred) #wyznaczenie średniego procentu błędu
        s+=mape
    return s/powtorzenia


print("Sredni błąd z zastąpieniem wartości ostających średnimi:",wielokrotne_testowanie_no_outliers_zastapienie_cech(5))
#Sredni błąd z zastąpieniem wartości ostających średnimi: 1.993626833871674


def generuj_wykres_wag():
    from sklearn.linear_model import LinearRegression #dołączenie bibliotek używanych w funkcji
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error
    X, y = tablica[:,:-1], tablica[:,-1]

    X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, shuffle=True) #podział na część testową i uczącą
    outliers = np.abs((y_train-y_train.mean())/y_train.std())>2 #wyznaczenie części odstającej

    #zastąpienie wartości odstających
    y_train_mean = y_train.copy()
    y_train_mean[outliers] = y_train.mean()


    linReg = LinearRegression() #wyznaczenie regresji liniowej
    linReg.fit(X_train, y_train_mean) #dopasowanie regresji do wartości uczących


    fig, ax = plt.subplots(1,1)
    x = np.arange(0,9)
    wagi = linReg.coef_
    ax.bar(x, wagi)
    ax.set_xticks(x)
    #ax.set_xticklabels(nazwy_kolumn, rotation = 90) # nie działają podpisy kolumn

#generuj_wykres_wag()

#Na podstawie wykresu da się zauważyć, że:
# - negatywnie na wyniki wpływa "total serum choresterol"[4]
# - pozytywnie na wyniki wpływa "bmi"[2], "low-density lipoproteins"[5], "blood sugar level"[9]

def wielokrotne_testowanie_no_outliers_zastapienie_cech(powtorzenia):
    from sklearn.linear_model import LinearRegression #dołączenie bibliotek używanych w funkcji
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error
    X, y = tablica[:,:-1], tablica[:,-1]

    s=0
    global nowe_dane
    nowe_dane = np.stack([ #wygenerowanie nowych danych
        X[:,4] / X[:,7],
        X[:,4] * X[:,3]
    ],axis=-1)
    global X_nowe
    X= np.concatenate([X,nowe_dane],axis=-1) #dodanie nowych kolumn z danymi do zbioru danych

    for a in range(powtorzenia):
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, shuffle=True) #podział na część testową i uczącą
        outliers = np.abs((y_train-y_train.mean())/y_train.std())>2 #wyznaczenie części odstającej

        #zastąpienie wartości odstających
        y_train_mean = y_train.copy()
        y_train_mean[outliers] = y_train.mean()

        linReg = LinearRegression() #wyznaczenie regresji liniowej
        linReg.fit(X_train, y_train_mean) #dopasowanie regresji do wartości uczących
        y_pred = linReg.predict(X_test) #przetestowanie nauczonego modelu regresji
        mape = mean_absolute_percentage_error (y_test, y_pred) #wyznaczenie średniego procentu błędu
        s+=mape
    return s/powtorzenia

print("Sredni błąd z dodaniem nowych zestawów danych:",wielokrotne_testowanie_no_outliers_zastapienie_cech(5))
#Sredni błąd z dodaniem nowych zestawów danych: 1.747935536944405

#2.1 - wygenerowanie macierzy korelacji
#    - wykresy korelacji między wartościami niezależnymi a wartością zależną
#2.2 - testowanie modelu regresji liniowej  (wielokrotne w funkcji) + średni procent błędu regresji
#2.3 - usuwanie wartości odstających
#2.4 - generowanie nowych cech

#%%

"""
 :Attribute Information:
##       - age     age in years [0]
##       - sex      [1]
##       - bmi     body mass index [2]
##       - bp      average blood pressure [3]
##       - s1      tc, total serum cholesterol [4]
##       - s2      ldl, low-density lipoproteins [5]
##       - s3      hdl, high-density lipoproteins [6]
##       - s4      tch, total cholesterol / HDL [7]
##       - s5      ltg, possibly log of serum triglycerides level [8]
##       - s6      glu, blood sugar level [9]
"""
