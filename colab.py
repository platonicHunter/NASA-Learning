import pandas as pd

data = {
    "Item": ["Apple", "Banana", "Orange", "Grapes"],
    "Quantity": [10, 20, 15, 12],
    "Price": [100, 200, 150, 120]
}

df = pd.DataFrame(data)

df["Amount"] = df["Quantity"]*df["Price"]

print(df)
