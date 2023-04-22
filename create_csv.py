import numpy as np
import pandas as pd

# 顧客属性のデータを生成する
n_customers = 10000
age = np.random.randint(18, 80, n_customers)
gender = np.random.choice(["Male", "Female"], n_customers)
location = np.random.choice(["Tokyo", "Osaka", "Kyoto", "Fukuoka"], n_customers)
occupation = np.random.choice(
    ["Student", "Office worker", "Freelancer", "Retired"], n_customers
)
family_size = np.random.randint(1, 6, n_customers)
income = np.random.normal(500000, 100000, n_customers)

# 購入履歴のデータを生成する
n_purchases = 50000
purchase_history = np.random.choice(["Yes", "No"], n_purchases, p=[0.6, 0.4])
product_category = np.random.choice(
    ["Electronics", "Clothing", "Food", "Books"], n_purchases
)
brand = np.random.choice(["Sony", "Nike", "Nestle", "Amazon"], n_purchases)
price = np.random.normal(5000, 1000, n_purchases)
features = np.random.randint(1, 6, n_purchases)
reviews = np.random.randint(1, 6, n_purchases)
purchase_date = pd.date_range("2022-01-01", periods=n_purchases, freq="D")
purchase_amount = np.random.normal(10000, 2000, n_purchases)
purchase_location = np.random.choice(
    ["Tokyo", "Osaka", "Kyoto", "Fukuoka"], n_purchases
)
payment_method = np.random.choice(
    ["Credit card", "Cash on delivery", "Bank transfer"], n_purchases
)

# データフレームに変換する
df_customer = pd.DataFrame(
    {
        "age": age,
        "gender": gender,
        "location": location,
        "occupation": occupation,
        "family_size": family_size,
        "income": income,
    }
)
df_purchase = pd.DataFrame(
    {
        "purchase_history": purchase_history,
        "product_category": product_category,
        "brand": brand,
        "price": price,
        "features": features,
        "reviews": reviews,
        "purchase_date": purchase_date,
        "purchase_amount": purchase_amount,
        "purchase_location": purchase_location,
        "payment_method": payment_method,
    }
)

# 顧客IDと購入IDを追加する
df_customer["customer_id"] = range(1, n_customers + 1)
df_purchase["purchase_id"] = range(1, n_purchases + 1)

# 購入履歴に顧客IDを紐付ける
df_purchase["customer_id"] = np.random.randint(1, n_customers + 1, n_purchases)

# データセットを作成する
df = pd.merge(df_purchase, df_customer, on="customer_id", how="left")

# CSVファイルにデータセットを保存する
df.to_csv("customers_df.csv", index=False)
