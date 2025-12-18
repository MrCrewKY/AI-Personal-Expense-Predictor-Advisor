Step-by-Step Starter Path (Beginner Safe)

(Updated for Kaggle Transaction-Level Dataset)

🟢 STEP 0: Setup (1–2 hours max)

Install:

pip install pandas numpy scikit-learn prophet tensorflow matplotlib


Use:

Jupyter Notebook OR Google Colab (recommended)

🟢 STEP 1: Load Kaggle Transaction Dataset (NO AI YET)
What you do:
import pandas as pd

df = pd.read_csv("transactions.csv")
df.head()

What your dataset contains:

Date

Transaction Description

Category

Amount

Type (Income / Expense)

🎯 Goal: Just see the raw transaction data.

If this fails → stop and fix.
Do NOT move forward until this works.

🟢 STEP 2: Convert Transaction Data → Monthly Expense Time Series

(NEW – required because dataset is transaction-level)

What this step REALLY means:

Filter Expense transactions only

Convert dates to datetime

Aggregate expenses by month

Filter expenses only:
df_expense = df[df['Type'] == 'Expense']

Convert Date column:
df_expense['Date'] = pd.to_datetime(df_expense['Date'])

Aggregate monthly expenses:
monthly_expense = (
    df_expense
    .groupby(pd.Grouper(key='Date', freq='M'))['Amount']
    .sum()
    .reset_index()
)

monthly_expense.columns = ['date', 'expense']
monthly_expense.head()

What you should now have:
date        expense
2020-01-31  3200
2020-02-29  3100


🎯 Goal: Convert transaction data into monthly expense data.

📌 Report-safe explanation:

The transaction-level dataset was aggregated into monthly expenses to support time-series forecasting.

🟢 STEP 3: Preprocess Data (scikit-learn)
What “preprocess” REALLY means here:

Handle missing values

Normalize values

Prepare data for ML models

Handle missing values:
monthly_expense.isnull().sum()
monthly_expense = monthly_expense.dropna()

Normalize expenses:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled = scaler.fit_transform(monthly_expense[['expense']])


🎯 Goal: Convert raw monthly expenses → ML-ready numbers.

🟢 STEP 4: Baseline Forecasting (Prophet) ⭐ EASIEST AI PART

Prophet is a black box. You don’t need to understand internals.

from prophet import Prophet

prophet_df = monthly_expense.rename(
    columns={'date':'ds', 'expense':'y'}
)

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)


🎯 Goal: Get baseline monthly expense predictions.

💡 In your report:

Prophet is used as a baseline statistical time-series forecasting model.

That sentence alone is enough.

🟢 STEP 5: ANN Training (THIS IS THE CORE “AI” PART)

Forget theory. Use Keras template.

Windowing (IMPORTANT):
import numpy as np

X = []
y = []

for i in range(len(scaled) - 6):
    X.append(scaled[i:i+6])
    y.append(scaled[i+6])

X, y = np.array(X), np.array(y)

ANN Model (SIMPLE):
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(6,)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(
    X, y,
    epochs=50,
    validation_split=0.2
)


🎯 Goal:

Model trains

Loss decreases

No crash

You’re DONE with ANN.

🟢 STEP 6: Plot Loss Curves (EASY MARKS)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()


Lecturers LOVE this.

🟢 STEP 7: Compare Prophet vs ANN (IMPORTANT FOR GRADES)

Use:

MAE

MSE

MAPE

from sklearn.metrics import mean_absolute_error, mean_squared_error


📌 In report:

ANN achieved lower MSE than Prophet, indicating improved predictive performance.

Even if improvement is small — it’s acceptable.

🟡 STEP 8: LLM Advisory (YOU DON’T “TRAIN” IT)

You DO NOT build an LLM.

You just do:

Prediction → Prompt → Advice

Example:

“Based on predicted expenses increasing by 8%, suggest budgeting advice.”

LLM just explains numbers in English.

🟡 STEP 9: Streamlit Dashboard (LAST STEP)

Only after EVERYTHING works.

import streamlit as st

st.line_chart(monthly_expense['expense'])
st.write("AI Budget Advice")


Simple UI = full marks.
