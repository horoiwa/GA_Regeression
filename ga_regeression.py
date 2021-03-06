import pandas as pd
from sklearn.datasets import load_boston
from multi_objective import RidgeGA


def load_dataset():
    X = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
    y = load_boston().target
    return X, y 


def polynomial_features(dataframe):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2)
    poly.fit(dataframe)
    dataframe_poly = pd.DataFrame(poly.transform(dataframe),
                                  columns=poly.get_feature_names(input_features=dataframe.columns)) 
    return dataframe_poly


def standard_scaler(dataframe):
    from sklearn.preprocessing import StandardScaler
    stsc = StandardScaler()
    stsc.fit(dataframe)
    dataframe_sc = pd.DataFrame(stsc.transform(dataframe),
                                columns=dataframe.columns) 
    return dataframe_sc


def run_ridge_ga(X, y):
    ridge_ga = RidgeGA(X, y, 10)
    ridge_ga.run()
    result = ridge_ga.result
    print(result)


if __name__ == '__main__':
    X, y = load_dataset()
    X_poly = polynomial_features(X)
    X_poly_sc = standard_scaler(X_poly)
    run_ridge_ga(X, y)
