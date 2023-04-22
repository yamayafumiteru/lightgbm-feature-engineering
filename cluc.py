import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# データセットを読み込む
df = pd.read_csv("dataset.csv")

# カテゴリカル変数に変換する列のリスト
categorical_columns = [
    "gender",
    "location",
    "occupation",
    "purchase_history",
    "product_category",
    "brand",
    "purchase_location",
    "payment_method",
]

# カテゴリカル変数に変換する
for col in categorical_columns:
    df[col] = df[col].astype("category")

# 特徴量を選択する
features = [
    "age",
    "gender",
    "location",
    "occupation",
    "family_size",
    "income",
    "purchase_history",
    "product_category",
    "brand",
    "price",
    "features",
    "reviews",
    "purchase_location",
    "payment_method",
]

# 特徴量とターゲットに分割する
X = df[features]
y = df["purchase_amount"]

# トレーニングセットとテストセットに分割する
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LightGBMのハイパーパラメータを設定する
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}
# データセットを作成する
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# モデルをトレーニングする
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    early_stopping_rounds=10,
)

# テストセットを使用してモデルの性能を評価する
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
