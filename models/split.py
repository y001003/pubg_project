import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import preprocessing as p

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_val, y_train, y_val = train_test_split(p.X_train,p.y_train,test_size=0.2, random_state=42)


# 표준정규분포로
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)