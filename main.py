from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import pandas as pd

# scaler = StandardScaler()

basic_train = pd.read_csv('BasicDataset_Training_MRN.csv')
basic_test = pd.read_csv('BasicDataset_Test_MRN.csv')

train = basic_train.drop(['User_Satisfaction','Unnamed: 0','Unnamed: 0.1'],axis = 1)
test = basic_test.drop(['User_Satisfaction','Unnamed: 0','Unnamed: 0.1'],axis = 1)
ground_truth_train = basic_train.loc[:,'User_Satisfaction'].copy()
ground_truth_test = basic_test.loc[:,'User_Satisfaction'].copy()

# Yeo Johnson Transformation, make it closed to Gaussian distribution
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=True,)

train = pt.fit_transform(train)
test = pt.fit_transform(test)

# seems like no need to split the validation dataset when I did the test
# train = scaler.fit_transform(train)
# x_train, x_val, y_train, y_val = train_test_split(train,ground_truth_train, test_size=0.2)

# establish Keras model
model = Sequential()

model.add(Dense(20, activation='relu', input_shape=(len(train),12)))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#
# callbacks_list = [
#     keras.callbacks.ModelCheckpoint(
#         filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
#         monitor='val_loss', save_best_only=True),
#     keras.callbacks.EarlyStopping(monitor='acc', patience=1)]
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=["accuracy"])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train,ground_truth_train,batch_size=50,validation_data=(test,ground_truth_test),epochs=30,verbose=1)

# from keras.models import load_model
# model=load_model('best_model.17-0.60.h5')

# evaluate the model
score=model.evaluate(test,ground_truth_test,verbose=2)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

#prepare for the AUC-ROC curve
from sklearn.metrics import roc_curve
y_pred_keras = model.predict(test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(ground_truth_test, y_pred_keras)

from sklearn.ensemble import RandomForestClassifier
# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=10)
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
