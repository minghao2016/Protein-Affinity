#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:20:07 2019

@author: ambuj
"""

da_log = X_std.columns[['DA_' in x for x in X_std.columns]]
df_log.shape
df_log = pcas_leftpca[pcas_leftpca.columns[['log_' in x for x in pcas_leftpca.columns]]]
df_log.columns
#df_log.drop(['log_IVLFCMA_pair', 'log_GTSWYPHEQDNKR_pair'], axis=1, inplace=True)
df_log.columns
pca = PCA(n_components = 5)
pca.fit_transform(df_log)
log_pca1 = pca.fit_transform(df_log)
df_log_pca2 = pd.DataFrame()
log_pca1[0]
log_pca1[:,0]
df_log_pca2['log_pca1'] = log_pca1[:,0]
df_log_pca2['log_pca2'] = log_pca1[:,1]
df_log_pca2['log_pca3'] = log_pca1[:,2]
df_log_pca2['log_pca4'] = log_pca1[:,3]
df_log_pca2['log_pca5'] = log_pca1[:,4]
df_log_pca2['delG'] = X_std['delG']