#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:36:13 2019

@author: ambuj
"""
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
import math
from sklearn.model_selection import cross_val_predict, GridSearchCV
import random
from sklearn.svm import SVR


def rfr_model(X, y):

# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,6),
            'n_estimators': (10, 50, 100),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],                               random_state=False, verbose=False)

# Perform K-Fold CV
#    scores = cross_val_predict(rfr, X, y, cv=10)
#    
#    rval2 = np.corrcoef(scores, y)
#    print rval2
    return rfr



def support_regression (df2, results):
#    svr_lin = SVR(kernel='linear', C=1000, gamma='auto')
#    svr_lin.fit(df2, results)
#    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=6, epsilon=.1, coef0=1)
    return svr_poly
def ridge_regression (df2, results):
    rlm = Ridge(alpha=1)
    model = rlm.fit(df2, results)
    return model
def lasso_regression (df2, results):
    rlm = linear_model.Lasso(alpha=1)
    model = rlm.fit(df2, results)
    return model
def logistic_regression (df2, results):
    rf_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
    model = rf_lm.fit(df2, results)
    return model

def decision_tree(df2, results):
    dt = tree.DecisionTreeClassifier(criterion='gini')
    model = dt.fit(df2, results)
#    dt.score(df2,results)
#    
#    preds = dt.predict(df2)
    return model

def linear_regression(df2, results):
    lm = linear_model.LinearRegression()
    model = lm.fit(df2, results)

#    preds = gnb.predict(df2)
    return model

def frwd_selection(df2):
    
    newdf = pd.DataFrame()
    results = df2['delG'].fillna(0).values
    df2 = df2.drop(['delG'],axis=1)
    col_list  = list(df2.columns)
    random.shuffle(col_list)
    corr2 = 0 
    for c in col_list:
        newdf[c] = df2[c]
        X = newdf.fillna(0).values
        Y = results
        model = support_regression(newdf,Y)
        cv_results2 = cross_val_predict(model,X,Y, cv=10) 
        rval2 = np.corrcoef(Y, cv_results2)
        if rval2[0][1] > corr2:
            corr2 = rval2[0][1]
        else:
            newdf.drop(c,axis=1,inplace=True)
#    print corr2
    newdf['delG'] = results
    return newdf

res_df = pd.DataFrame()
res_dict = {}
for i in range (1000):
    print i
    cols = ['delG']
    datax = norm_fe.fillna(0)
    
    #datax['delG'] = [-math.exp(abs(x)) for x in ndf6['delG']]
    newdata = frwd_selection(datax)
    datax = newdata
    names = datax.columns
    X = datax.drop(cols, axis=1).fillna(0).values
    Y = datax['delG'].fillna(0).values
    
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    X_train = X
    X_test = X
    y_train = Y
    y_test = Y
    
    transformer = PolynomialFeatures(degree=1, include_bias=False)
    tX_train = transformer.fit_transform(X_train)
    #==============================================================================
    model = support_regression(tX_train,y_train)
    rfe = RFE(model, len(datax.columns)-1)
    #rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2), scoring='r2')
    fit = rfe.fit(tX_train, y_train)
    #==============================================================================
    
    
    Y_predict = fit.predict(X_test)
    mae = mean_absolute_error(y_test, Y_predict)
    rval = np.corrcoef(y_test, Y_predict)
#    print mae,rval
    ss = ShuffleSplit(n_splits=10, test_size=0.25,random_state=0)
    #for train_index, test_index in ss.split(X):
    #    print("%s %s" % (train_index, test_index))
    
    
    
    
    #========================================================================
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    cv_results1 = cross_val_score(fit,tX_train,y_train, cv=cv) 
    cv_results2 = cross_val_predict(fit,tX_train,y_train, cv=10) 
    #========================================================================
    rval2 = np.corrcoef(y_test, cv_results2)
    mae2 = mean_absolute_error(y_test, cv_results2)
#    print mae2, rval2
    #====================PLOT===================================================#
#    fig,ax = plt.subplots()
#    ax.scatter(y_test, cv_results2)
#    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
#    ax.set_xlabel('Measured')
#    ax.set_ylabel('Predicted')
#    fig.show()
    #===========================================================================#
    #X_new = X_[:,fit.support_]
    #====================PRINT===============================================
    #print "Features sorted by their rank:"
    imp_names = []
    for i, j in sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)):
        if i == 1:
            imp_names.append(j)
#    print imp_names
    res_dict.update({'features':imp_names, 'corr': rval[0][1],'corr_val2': rval2[0][1]})
    
    res_df = res_df.append(res_dict, ignore_index=True)
#r_sq = model.score(X_, Y)
#print('coefficient of determination:', r_sq)
##print('intercept:', model.intercept_)
##print('coefficients:', model.coef_)
#print("Num Features: %d") % fit.n_features_
#print("Selected Features: %s") % fit.support_
#print("Feature Ranking: %s") % fit.ranking_
#========================================================================