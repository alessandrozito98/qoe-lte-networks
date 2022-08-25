from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

basic_train = pd.read_csv('BasicDataset_Training_MRN.csv')
basic_test = pd.read_csv('BasicDataset_Test_MRN.csv')

new_feature_train = pd.DataFrame(data=(basic_train.Cumulative_Full_Service_Time_LTE/(basic_train.Cumulative_Full_Service_Time_LTE+basic_train.Cumulative_Lim_Service_Time_LTE+basic_train.Cumulative_No_Service_Time_LTE)).round(1),columns=['new_feature'])
new_feature_train['new_feature'].fillna(0,inplace=True)
new_feature_test = pd.DataFrame(data=(basic_test.Cumulative_Full_Service_Time_LTE/(basic_test.Cumulative_Full_Service_Time_LTE+basic_test.Cumulative_Lim_Service_Time_LTE+basic_test.Cumulative_No_Service_Time_LTE)).round(1),columns=['new_feature'])
new_feature_test['new_feature'].fillna(0,inplace=True)

# create new feature CDF
# cdf_train = basic_train.Cumulative_Lim_Service_Time_LTE
# cdf_test = basic_train.loc[:, 'User_Satisfaction'].copy()
# yt = cdf_test.copy()
# bin_edges = np.linspace(min(cdf_train), max(cmf_train), 101)
# neg, _ = np.histogram(cdf_train[yt == +1], bins=bin_edges)  # count number of evidences per bin
# pos, _ = np.histogram(cdf_train[yt == 0], bins=bin_edges)
# sumpos =  sum(pos)
# sumneg =  sum(neg)
# pos = pos.astype(float) / sumpos  # normalize to total number of evidences
# neg = neg.astype(float) / sumneg
# xrange = bin_edges[1:] - bin_edges[:1]
# plt.plot(xrange,np.cumsum(pos))
# plt.plot(xrange,np.cumsum(neg))
# plt.show()
#
train = basic_train.drop(['User_Satisfaction','Unnamed: 0','Unnamed: 0.1'],axis = 1)
train = train.join(new_feature_train,how='right')
test = basic_test.drop(['User_Satisfaction','Unnamed: 0','Unnamed: 0.1'],axis = 1)
test = test.join(new_feature_test,how='right')
ground_truth_train = basic_train.loc[:,'User_Satisfaction'].copy()
ground_truth_test = basic_test.loc[:,'User_Satisfaction'].copy()


# Yeo Johnson Transformation, make it closed to Gaussian distribution
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=True,)
#
train = pt.fit_transform(train)
test = pt.fit_transform(test)

# Plot Distribution of Cum. Full Service Time in LTE (no used)
# fig, ax = plt.subplots(1, 1, figsize=(20, 5))
# x = basic_train.Cumulative_Full_Service_Time_UMTS
# tmp_0 = ax.hist(x, bins=100)
# ax.set_ylabel('Bincount')
# ax.xaxis.label.set_color('red')
# ax.yaxis.label.set_color('red')
# ax.tick_params(axis='x', colors='red')
# ax.tick_params(axis='y', colors='red')
# plt.show()

x_train, x_val, y_train, y_val = train_test_split(train,ground_truth_train, test_size=0.2)
# # establish Keras model
model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(len(train),13)))
model.add(Dense(32,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=10,validation_data=(x_val,y_val),epochs=30,verbose=1)

# print(model.summary())(no used)
# # from keras.models import load_model
# # model=load_model('best_model.17-0.60.h5')
#
# # evaluate the model
# score=model.evaluate(test,ground_truth_test,verbose=2)
# print('Test loss:',score[0])
# print('Test accuracy:',score[1])
#
#prepare for the AUC-ROC curve(used)
from sklearn.metrics import roc_curve
y_pred_keras = model.predict(test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(ground_truth_test, y_pred_keras)

# Supervised transformation based on random forests
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=10, n_estimators=40)
rf.fit(train, ground_truth_train)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

y_pred_rf = rf.predict_proba(test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(ground_truth_test, y_pred_rf)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
