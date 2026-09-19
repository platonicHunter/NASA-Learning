import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Year": [
        2000, 2001, 2002, 2003, 2004, 2005,
        2006, 2007, 2008, 2009, 2010, 2011,
        2012, 2013, 2014, 2015, 2016, 2017,
        2018, 2019, 2020, 2021, 2022, 2023,
        2024, 2025
    ],
    "GMSL": [
        18.70, 23.80, 25.91, 28.93, 30.62, 34.65,
        36.14, 37.35, 40.95, 44.45, 46.25, 45.51,
        55.32, 57.91, 60.42, 69.26, 73.33, 73.80,
        76.24, 82.73, 85.21, 89.60, 91.68, 97.03,
        100.68, 100.96
    ]
}


df = pd.DataFrame(data)

print("NASA Sea Level Data")
print(df)

plt.figure(figsize=(10, 5))

plt.scatter(df["Year"], df["GMSL"], color="blue")

plt.xlabel("Year")
plt.ylabel("Annual Mean GMSL (mm)")
plt.title("Annual Mean Global Mean Sea Level")

plt.grid(True)
plt.show()


# Linear Regression
x = df[["Year"]]
y = df["GMSL"]

model = LinearRegression()
model.fit(x, y)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)


# Prediction
df["Predicted"] = model.predict(x)
plt.figure(figsize=(10, 5))

plt.scatter(
    df["Year"],
    df["GMSL"],
    color="blue",
    label="Actual Data"
)

plt.plot(
    df["Year"],
    df["Predicted"],
    color="red",
    label="Linear Regression"
)

plt.xlabel("Year")
plt.ylabel("Annual Mean GMSL (mm)")
plt.title("NASA Sea Level Data with Linear Regression")

plt.legend()
plt.grid(True)
plt.show()


# Predic GMSL for 2030, 2040, 2050

futureyear = pd.DataFrame({"Year": [2030, 2040, 2050]})
futureyear["Predicted GMSL"] = model.predict(futureyear)
print(futureyear)

# actual data and future predictions together

plt.figure(figsize=(10, 5))

# Actual data and linear regression
plt.scatter(
    df["Year"],
    df["GMSL"],
    color="blue",
    label="Actual Data"
)
# Linear regression line
plt.plot(
    df["Year"],
    df["Predicted"],
    color="red",
    label="Linear Regression"
)
# Future predictions
plt.scatter(
    futureyear["Year"],
    futureyear["Predicted GMSL"],
    color="green",
    label="Future Predictions"
)

plt.xlabel("Year")
plt.ylabel("Annual Mean GMSL (mm)")
plt.title("NASA Sea Level Data with Linear Regression")

plt.legend()
plt.grid(True)
plt.show()
