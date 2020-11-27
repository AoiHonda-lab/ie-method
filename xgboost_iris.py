import xgboost as xgb
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, mean_squared_error
import sklearn.preprocessing as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data)
iris_df = iris_df.rename(columns={
    0: 'sepal_length',
    1: 'sepal_width',
    2: 'petal_length',
    3: 'petal_width'})
iris_df['target'] = iris.target

# 数字のカテゴリをカテゴリ名に埋めなおしている
for i, name in enumerate(iris.target_names):
    iris_df['target'] = iris_df['target'].where(iris_df['target'] != i, name)

# ラベルエンコーダを使ってまた数字に戻している
le = sp.LabelEncoder()
le.fit(iris_df.target.unique())
iris_df.target = le.fit_transform(iris_df.target)
    
# OneHotEncoderでtargetを3つのクラスのOneHot表現に変更している
ohe = sp.OneHotEncoder()
enced = ohe.fit_transform(iris_df.target.values.reshape(1, -1).transpose())
temp = pd.DataFrame(index=iris_df.target.index, columns="target-" + le.classes_, data=enced.toarray())
iris_df = pd.concat([iris_df, temp], axis=1)
del iris_df['target']

iris_df.head()

train_df, test_df = model_selection.train_test_split(iris_df, test_size=0.3)
train_df_y = train_df[['sepal_length']]
train_df_x = train_df.copy().drop('sepal_length', axis=1)
test_df_y = test_df[['sepal_length']]
test_df_x = test_df.copy().drop('sepal_length', axis=1)

clf = xgb.XGBRegressor()

# ハイパーパラメータ探索
clf_cv = model_selection.GridSearchCV(clf, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
clf_cv.fit(train_df_x, [i[0] for i in train_df_y.values])
print(clf_cv.best_params_, clf_cv.best_score_)

# 改めて最適パラメータで学習
clf = xgb.XGBRegressor(**clf_cv.best_params_)
clf.fit(train_df_x, [i[0] for i in train_df_y.values])

mean_pred = [train_df_y.mean() for i in range(len(test_df_y))]
rmse_base = np.sqrt(mean_squared_error(test_df_y, mean_pred))
print("学習データの平均を予測としたやつをBaseLineとする\nBaseLineのrmse: " + str(rmse_base))

pred = clf.predict(test_df_x)
rmse = np.sqrt(mean_squared_error(test_df_y, pred))
print("予測したやつのrmse: " + str(rmse))

plt.ylabel("Predict")
plt.xlabel("Actual")
plt.scatter(test_df_y, pred)
plt.plot([4.5, 8], [4.5, 8], c='r')
plt.show()

xgb.plot_importance(clf)
plt.show()