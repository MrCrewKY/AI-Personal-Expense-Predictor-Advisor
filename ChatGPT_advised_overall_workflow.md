Overall Workflow (Important)

You will use TWO TOOLS together:

Tool	Purpose
Google Colab	Learn, test, experiment, debug
Cursor AI	Write clean final code, refactor, fix errors

Think of it like:

Colab = laboratory
Cursor = workshop

🟢 STEP 0 — Setup (Google Colab)
✅ WHERE: Google Colab

1️⃣ Go to
👉 https://colab.research.google.com


Click New Notebook


2️⃣ Rename notebook:

expenses_forecasting.ipynb


3️⃣ Install libraries (Colab needs this once per session):

!pip install pandas numpy scikit-learn prophet tensorflow matplotlib


⛔ If this fails → STOP and fix
Do NOT move forward.

🎯 Goal: No red error text.

🟢 STEP 1 — Load Monthly Expense Data
✅ WHERE: Google Colab
Option A: Upload CSV manually

Left sidebar → 📁 → Upload → expenses.csv

Code cell:
import pandas as pd

df = pd.read_csv("expenses.csv")
df.head()


You should see something like:

date	expense
2021-01-01	1200
2021-02-01	1350

⛔ If you don’t see data → STOP

🎯 Goal: You can see rows of data.

🟢 STEP 2 — Data Preprocessing (scikit-learn)
✅ WHERE: Google Colab
2.1 Handle missing values
df.isnull().sum()


If missing:

df = df.dropna()

2.2 Normalize expenses
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_expense = scaler.fit_transform(df[['expense']])


🧠 What happened (plain English):

Numbers are rescaled so the ANN learns better.

2.3 Create sliding windows (6 → 1)
import numpy as np

X = []
y = []

for i in range(len(scaled_expense) - 6):
    X.append(scaled_expense[i:i+6])
    y.append(scaled_expense[i+6])

X = np.array(X)
y = np.array(y)


Quick sanity check:

X.shape, y.shape


🎯 Goal:

X is 2D (samples × 6)

y is 1D

🟢 STEP 3 — Prophet Baseline Forecast
✅ WHERE: Google Colab
from prophet import Prophet

prophet_df = df.rename(columns={'date': 'ds', 'expense': 'y'})
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

model_prophet = Prophet()
model_prophet.fit(prophet_df)

future = model_prophet.make_future_dataframe(periods=6, freq='M')
forecast = model_prophet.predict(future)


View result:

forecast[['ds', 'yhat']].tail()


🎯 Goal: Prophet produces future predictions.

📌 Report sentence (save this):

Prophet was used as a baseline statistical time-series forecasting model.

🟢 STEP 4 — ANN Training (TensorFlow)
✅ WHERE: Google Colab
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(6,)),
    Dense(32, activation='relu'),
    Dense(1)
])

ann_model.compile(
    optimizer='adam',
    loss='mse'
)

history = ann_model.fit(
    X, y,
    epochs=50,
    validation_split=0.2,
    verbose=1
)


🎯 Goal:

Training runs

Loss decreases

No crash

⛔ If it crashes → STOP and debug (Cursor helps here).

🟢 STEP 5 — Plot Training & Validation Loss
✅ WHERE: Google Colab
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()


🎓 Lecturer impression: 📈📈📈

🟢 STEP 6 — Compare Prophet vs ANN
✅ WHERE: Google Colab
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ANN prediction (last available window)
ann_pred = ann_model.predict(X)

mae_ann = mean_absolute_error(y, ann_pred)
mse_ann = mean_squared_error(y, ann_pred)

print("ANN MAE:", mae_ann)
print("ANN MSE:", mse_ann)


(Prophet MAE/MSE can be computed similarly.)

📌 Report line:

The ANN achieved a lower MSE than Prophet, indicating improved predictive performance.

🟡 STEP 7 — LLM Advisory (Conceptual, Simple)
✅ WHERE: Cursor AI (later)

Example prompt:

Given that next month’s predicted expense increased by 8% compared to the previous month,
generate budgeting advice for a university student.


You are NOT training the LLM.
You are using it to explain numbers.

🟡 STEP 8 — Streamlit Dashboard
✅ WHERE: Cursor AI

Only after everything works in Colab.

Cursor will help generate:

app.py

Charts

Text

Example:

import streamlit as st

st.title("AI Expense Forecasting")
st.line_chart(df['expense'])
st.write("AI Budget Advice")


Simple = full marks.

🧠 Where Cursor AI Fits (IMPORTANT)
Use Cursor to:

✔ Rewrite Colab code into .py files
✔ Fix errors
✔ Add comments
✔ Clean structure

Example Cursor prompt:
Refactor this ANN training code into a clean Python function and explain each step.

🧭 Final Advice (Very Honest)

If you:

Get Step 1–6 working in Colab

Understand what each block does

Use Cursor only to assist

👉 You are already ahead of many students.
