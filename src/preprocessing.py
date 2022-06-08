import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


url = './data/train_V2.csv'
url2 = './data/test_V2.csv'
url3 = './data/sample_submission_V2.csv'
data1 = pd.read_csv(url)
data2 = pd.read_csv(url2)
data3 = pd.read_csv(url3)
train_df = pd.DataFrame(data1)
test_df = pd.DataFrame(data2)
sample = pd.DataFrame(data3)

# 결측치 제거
train_df = train_df.loc[~train_df['winPlacePerc'].isnull()]

# 고유값을 나타내는 Id 값들 삭제
useless = ['Id','groupId','matchId']
train_select = train_df.loc[:,~train_df.columns.isin(useless)]
X_train = train_select.iloc[:,:-1]
y_train = train_select.iloc[:,-1]
X_train = X_train.drop(columns='matchType')

test_df = test_df.loc[:,~test_df.columns.isin(useless)]
X_test = test_df.drop(columns='matchType')

print(train_df.head())
