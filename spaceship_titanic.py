# %%
import sys
import numpy as np
import pandas as pd

import IPython
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

import scipy as sp

sns.set(font_scale=2.5)

import missingno as msno
import warnings
warnings.filterwarnings("ignore")

# %%
df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

# %%
# test 결측치 시각화
sns.heatmap(df_test.isnull(), cbar=False)

# %%
# test 데이터 결측치, 빈도 높은 다른 것으로 바꿔주기
most_freq = df_test['CryoSleep'].value_counts(dropna=True).idxmax()
df_test['CryoSleep'].fillna(most_freq, inplace = True)

most_freq = df_test['Age'].value_counts(dropna=True).idxmax()
df_test['Age'].fillna(most_freq, inplace = True)

most_freq = df_test['RoomService'].value_counts(dropna=True).idxmax()
df_test['RoomService'].fillna(most_freq, inplace = True)

most_freq = df_test['FoodCourt'].value_counts(dropna=True).idxmax()
df_test['FoodCourt'].fillna(most_freq, inplace = True)

most_freq = df_test['ShoppingMall'].value_counts(dropna=True).idxmax()
df_test['ShoppingMall'].fillna(most_freq, inplace = True)

most_freq = df_test['Spa'].value_counts(dropna=True).idxmax()
df_test['Spa'].fillna(most_freq, inplace = True)

most_freq = df_test['VRDeck'].value_counts(dropna=True).idxmax()
df_test['VRDeck'].fillna(most_freq, inplace = True)

most_freq = df_test['Destination'].value_counts(dropna=True).idxmax()
df_test['Destination'].fillna(most_freq, inplace = True)

most_freq = df_test['Cabin'].value_counts(dropna=True).idxmax()
df_test['Cabin'].fillna(most_freq, inplace = True)

most_freq = df_test['HomePlanet'].value_counts(dropna=True).idxmax()
df_test['HomePlanet'].fillna(most_freq, inplace = True)

print(df_test.loc[61])
print('\n')
print(df_test.loc[829])

# %%
df_train.head()

# %%
df_train.describe()

# %%
# 누락된 정보 확인
for col in df_train.columns:
    msg = 'colume: {:>10}`t Percent of NaN value: {:.2f}%'.format(col, 100*(df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msg)


# %%
# msno.matrix를 이용하여 행렬 형태로 데이터 보기
# figsize=(가로 길이, 세로 길이)
# color = (R, G, B)
msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.3, 0.5, 0.6))

# %%
# target label의 distribution 확인하기
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Transported'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Transported')
ax[0].set_ylabel('')
sns.countplot('Transported', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Transported')

plt.show()

# %%
# Age에 따른 Transported 그래프
fig, ax = plt.subplots(1, 1, figsize = (9, 5))
sns.kdeplot(df_train[df_train['Transported'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Transported'] == 0]['Age'], ax=ax)
plt.legend(['Transported == 1', 'Transported == 0'])
plt.show()
# 확인 결과 : 나이가 어릴수록 Transported 경우가 많음

# %%
cummulate_transported_ratio = []
for i in range(1, 80):
	cummulate_transported_ratio.append(df_train[df_train['Age'] < i]['Transported'].sum() / len(
    df_train[df_train['Age'] < i]['Transported']))

plt.figure(figsize=(7, 7))
plt.plot(cummulate_transported_ratio)
plt.title('Transported rate change depending on range of Age', y=1.02)
plt.xlabel('Range of Age(0~x)')
plt.ylabel('Transported rate')
plt.show()

# 확인 결과 : 나이가 어릴 수록 Transported 확률 높음, 중요한 feature임을 확인

# %%
# Destination에 따른 Transported 여부 확인
f,ax=plt.subplots(1,1,figsize=(7,6))
df_train[['Destination', 'Transported']].groupby(['Destination'], as_index=True).mean().sort_values(
	by='Transported', ascending=False).plot.bar(ax=ax)

# %%
 # 확인 결과 : Destinatio에 따른 Transported 차이가 있지만, 영향이 크지는 않을 것이라고 판단됨
 # 다른 feature와 비교
f,ax=plt.subplots(1,2,figsize=(20,15)) # 1x2 matrix

sns.countplot('Destination', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded')
sns.countplot('Destination', hue='Transported', data=df_train, ax=ax[1])
ax[1].set_title('(2) Destination vs Transported')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# 확인 결과 : 목적지가 TRAPPIST-1e인 승객이 가장 많았기때문에 Transported의 확률 또한 높을 수 밖에 없었음. 

# %%
# HomePlanet에 따른 Transported 여부 확인
f,ax=plt.subplots(1,1,figsize=(7,7))
df_train[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=True).mean().sort_values(
	by='Transported', ascending=False).plot.bar(ax=ax)

# %%
 # 확인 결과 : HomePlanet에 따른 Transported 차이가 있지만, 영향이 크지는 않을 것이라고 판단됨
 # 다른 feature와 비교
f,ax=plt.subplots(1,3,figsize=(30,20)) # 1x3 matrix

sns.countplot('HomePlanet', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded')
sns.countplot('HomePlanet', hue='Transported', data=df_train, ax=ax[1])
ax[1].set_title('(2) HomePlanet vs Transported')
sns.countplot('HomePlanet', hue='Destination', data=df_train, ax=ax[2])
ax[2].set_title('(3) HomePlanet vs Destination')


plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# 확인 결과 : HomePlanet이 Earth이면서, 목적지가 TRAPPIST-1e인 승객들의 Transported 비율이 높음

# %%
# RoomService에 따른 Transported 여부 확인
f,ax = plt.subplots(1,1,figsize=(8,8))
g = sns.distplot(df_train['RoomService'], color='b', label='Skewness : {:.2f}'.format(df_train['RoomService'].skew()), ax=ax)

# %%
# 확인 결과 : 한 쪽으로 치우쳐 있기 때문에 값에 log를 씌워 다시 확인
df_test.loc[df_test.RoomService.isnull(), 'RoomService'] = df_test['RoomService'].mean()

# testset에 있는 nan value를 평균값으로 치환

df_train['RoomService'] = df_train['RoomService'].map(lambda i : np.log(i) if i>0 else 0)
df_test['RoomService'] = df_test['RoomService'].map(lambda i : np.log(i) if i>0 else 0)

# %%
f, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['RoomService'], color='b', label='Skewness : {:.2f}'.format(
	df_train['RoomService'].skew()), ax=ax)
g = g.legend(loc='best')

# 확인 결과 : RoomService는 Transported 여부와 관련이 많지 않아 보임

# %%
# RoomService와 마찬가지로, FoodCourt, ShoppingMall, Spa and VRDeck 확인하기
# FoodCourt 확인
f,ax = plt.subplots(1,1,figsize=(8,8))
g = sns.distplot(df_train['FoodCourt'], color='b', label='Skewness : {:.2f}'.format(df_train['FoodCourt'].skew()), ax=ax)

# %%
# 확인 결과 : 한 쪽으로 치우쳐 있기 때문에 값에 log를 씌워 다시 확인
df_test.loc[df_test.FoodCourt.isnull(), 'FoodCourt'] = df_test['FoodCourt'].mean()

# testset에 있는 nan value를 평균값으로 치환

df_train['FoodCourt'] = df_train['FoodCourt'].map(lambda i : np.log(i) if i>0 else 0)
df_test['FoodCourt'] = df_test['FoodCourt'].map(lambda i : np.log(i) if i>0 else 0)

# %%
f, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['FoodCourt'], color='b', label='Skewness : {:.2f}'.format(
	df_train['FoodCourt'].skew()), ax=ax)
g = g.legend(loc='best')

# 확인 결과 : FoodCourt는 Transported 여부와 관련이 많지 않아 보임

# %%
# RoomService와 마찬가지로, FoodCourt, ShoppingMall, Spa and VRDeck 확인하기
# ShoppingMall 확인
f,ax = plt.subplots(1,1,figsize=(8,8))
g = sns.distplot(df_train['ShoppingMall'], color='b', label='Skewness : {:.2f}'.format(df_train['ShoppingMall'].skew()), ax=ax)

# %%
# 확인 결과 : 한 쪽으로 치우쳐 있기 때문에 값에 log를 씌워 다시 확인
df_test.loc[df_test.ShoppingMall.isnull(), 'ShoppingMall'] = df_test['ShoppingMall'].mean()

# testset에 있는 nan value를 평균값으로 치환

df_train['ShoppingMall'] = df_train['ShoppingMall'].map(lambda i : np.log(i) if i>0 else 0)
df_test['ShoppingMall'] = df_test['ShoppingMall'].map(lambda i : np.log(i) if i>0 else 0)

# %%
f, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['ShoppingMall'], color='b', label='Skewness : {:.2f}'.format(
	df_train['ShoppingMall'].skew()), ax=ax)
g = g.legend(loc='best')

# 확인 결과 : ShoppingMall는 Transported 여부와 관련이 많지 않아 보임

# %%
# RoomService와 마찬가지로, FoodCourt, ShoppingMall, Spa and VRDeck 확인하기
# Spa 확인
f,ax = plt.subplots(1,1,figsize=(8,8))
g = sns.distplot(df_train['Spa'], color='b', label='Skewness : {:.2f}'.format(df_train['Spa'].skew()), ax=ax)

# %%
# 확인 결과 : 한 쪽으로 치우쳐 있기 때문에 값에 log를 씌워 다시 확인
df_test.loc[df_test.Spa.isnull(), 'Spa'] = df_test['Spa'].mean()

# testset에 있는 nan value를 평균값으로 치환

df_train['Spa'] = df_train['Spa'].map(lambda i : np.log(i) if i>0 else 0)
df_test['Spa'] = df_test['Spa'].map(lambda i : np.log(i) if i>0 else 0)

# %%
f, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Spa'], color='b', label='Skewness : {:.2f}'.format(
	df_train['Spa'].skew()), ax=ax)
g = g.legend(loc='best')

# 확인 결과 : Spa가 높을 수록 Transported 확률이 높은 것 같아보이지만 대부분 이용하지 않았기때문에 Spa 또한 크게 영향을 미치지 않을 feature일 것이라고 판단

# %%
# RoomService와 마찬가지로, FoodCourt, ShoppingMall, Spa and VRDeck 확인하기
# VRDeck 확인
f,ax = plt.subplots(1,1,figsize=(8,8))
g = sns.distplot(df_train['VRDeck'], color='b', label='Skewness : {:.2f}'.format(df_train['VRDeck'].skew()), ax=ax)

# %%
# 확인 결과 : 한 쪽으로 치우쳐 있기 때문에 값에 log를 씌워 다시 확인
df_test.loc[df_test.VRDeck.isnull(), 'VRDeck'] = df_test['VRDeck'].mean()

# testset에 있는 nan value를 평균값으로 치환

df_train['VRDeck'] = df_train['VRDeck'].map(lambda i : np.log(i) if i>0 else 0)
df_test['VRDeck'] = df_test['VRDeck'].map(lambda i : np.log(i) if i>0 else 0)

# %%
f, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['VRDeck'], color='b', label='Skewness : {:.2f}'.format(
	df_train['VRDeck'].skew()), ax=ax)
g = g.legend(loc='best')

# 확인 결과 : VRDeck는 Transported 여부와 관련이 많지 않아 보임

# %%
# CryoSleep에 따른 Transported 여부 확인
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train[['CryoSleep','Transported']].groupby(['CryoSleep'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Transported vs CryoSleep')
sns.countplot('CryoSleep', hue='Transported', data=df_train, ax=ax[1])
ax[1].set_title('CryoSleep: Transported vs Untransported')
plt.show()

# 확인 결과 : CryoSleep을 True한 사람들이 Transported 확률이 높음, 모델에 영향이 있을 것이라고 판단

# %%
# 관련없는 PassengerId, Name, VIP 삭제, 다양한 값을 가진 값 삭제
df_train.drop(['PassengerId', 'HomePlanet','Destination', 'Name', 'Cabin', 'VIP'], axis = 1, inplace=True)
df_test.drop(['PassengerId', 'HomePlanet','Destination', 'Name', 'Cabin', 'VIP'], axis = 1, inplace=True)

# %%
df_train.head()

# %%
df_test.head()

# %%
#importing all the required ML packages
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics # 모델 평가를 위함
from sklearn.model_selection import train_test_split # training set을 쉽게 나눠주는 함수

# %%
# 결측치 시각화
sns.heatmap(df_train.isnull(), cbar=False)

# %%
# 데이터 결측치, 빈도 높은 다른 것으로 바꿔주기
most_freq = df_train['CryoSleep'].value_counts(dropna=True).idxmax()
df_train['CryoSleep'].fillna(most_freq, inplace = True)

most_freq = df_train['Age'].value_counts(dropna=True).idxmax()
df_train['Age'].fillna(most_freq, inplace = True)

most_freq = df_train['RoomService'].value_counts(dropna=True).idxmax()
df_train['RoomService'].fillna(most_freq, inplace = True)

most_freq = df_train['FoodCourt'].value_counts(dropna=True).idxmax()
df_train['FoodCourt'].fillna(most_freq, inplace = True)

most_freq = df_train['ShoppingMall'].value_counts(dropna=True).idxmax()
df_train['ShoppingMall'].fillna(most_freq, inplace = True)

most_freq = df_train['Spa'].value_counts(dropna=True).idxmax()
df_train['Spa'].fillna(most_freq, inplace = True)

most_freq = df_train['VRDeck'].value_counts(dropna=True).idxmax()
df_train['VRDeck'].fillna(most_freq, inplace = True)

print(df_train.loc[61])
print('\n')
print(df_train.loc[829])

# %%
# 결측치 시각화
sns.heatmap(df_train.isnull(), cbar=False)

# %%
X_train = df_train.drop('Transported', axis=1).values
# X_train에 'Transported'을 제외한 나머지들을 전부 values로 넣기, value를 처리하면 array 형태로 변환됨
target_label = df_train['Transported'].values
# target_label에 'Transported' values의 값만 넣는 것입니다.
X_test = df_test.values

# %%
# validation set 만들기
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)

# %%
model = RandomForestClassifier()
model.fit(X_tr, y_tr)
prediction = model.predict(X_vld)
print('총 {}명 중 {:.2f}% 정확도'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))

# %%
dataframe = pd.read_csv('./test.csv')

# %%
dataframe.head()

# %%
data = model.predict(X_test)
dataframe['Transported'] = data

# %%
dataframe.to_csv('1912943.csv', index=False)
