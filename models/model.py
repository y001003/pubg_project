import models.split as s

from sklearn.linear_model import LinearRegression# 1. Linear Regression
from sklearn.linear_model import Lasso# 2. Lasso
from sklearn.linear_model import Ridge# 3. Ridge
from xgboost.sklearn import XGBRegressor# 4. XGBoost
from lightgbm.sklearn import LGBMRegressor# 5. LightGBM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression# 1. Linear Regression
from sklearn.linear_model import Lasso# 2. Lasso
from sklearn.linear_model import Ridge# 3. Ridge
from xgboost.sklearn import XGBRegressor# 4. XGBoost
from lightgbm.sklearn import LGBMRegressor# 5. LightGBM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# training
reg = LinearRegression()
reg2 = Lasso()
reg3 = Ridge()
reg4 = XGBRegressor()
reg5 = LGBMRegressor()

reg.fit(X_train, y_train)
reg2.fit(X_train, y_train)
reg3.fit(X_train, y_train)
reg4.fit(X_train, y_train)
reg5.fit(X_train, y_train)

pred_train = reg.predict(X_train)
pred_train2 = reg2.predict(X_train)
pred_train3 = reg3.predict(X_train)
pred_train4 = reg4.predict(X_train)
pred_train5 = reg5.predict(X_train)

pred_val = reg.predict(X_val)
pred_val2 = reg2.predict(X_val)
pred_val3 = reg3.predict(X_val)
pred_val4 = reg4.predict(X_val)
pred_val5 = reg5.predict(X_val)

mse_train = mean_squared_error(y_train, pred_train)
mse_val = mean_squared_error(y_val, pred_val)
mse_train2 = mean_squared_error(y_train, pred_train2)
mse_val2 = mean_squared_error(y_val, pred_val2)
mse_train3 = mean_squared_error(y_train, pred_train3)
mse_val3 = mean_squared_error(y_val, pred_val3)
mse_train4 = mean_squared_error(y_train, pred_train4)
mse_val4 = mean_squared_error(y_val, pred_val4)
mse_train5 = mean_squared_error(y_train, pred_train5)
mse_val5 = mean_squared_error(y_val, pred_val5)

rscore_train = r2_score(y_train, pred_train)
rscore_val = r2_score(y_val, pred_val)
rscore_train2 = r2_score(y_train, pred_train2)
rscore_val2 = r2_score(y_val, pred_val2)
rscore_train3 = r2_score(y_train, pred_train3)
rscore_val3 = r2_score(y_val, pred_val3)
rscore_train4 = r2_score(y_train, pred_train4)
rscore_val4 = r2_score(y_val, pred_val4)
rscore_train5 = r2_score(y_train, pred_train5)