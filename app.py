import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from prophet import Prophet # <--- NEW IMPORT

# ==========================================
# 1. SETUP & UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI Expense Advisor", page_icon="💰", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .metric-card { background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #4b5563; text-align: center; }
    .advice-box-warn { background-color: rgba(255, 75, 75, 0.1); border-left: 5px solid #ff4b4b; padding: 15px; border-radius: 5px; }
    .advice-box-good { background-color: rgba(0, 217, 165, 0.1); border-left: 5px solid #00d9a5; padding: 15px; border-radius: 5px; }
    h1, h2, h3 { color: #feca57; font-family: sans-serif; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR: MODEL SELECTION
with st.sidebar:
    st.header("⚙️ Model Settings")
    model_choice = st.radio(
        "Choose Prediction Model:",
        ("ANN (Neural Network)", "Facebook Prophet")
    )
    st.info("""
    **ANN:** Good for capturing complex patterns from recent history (last 6 months).
    
    **Prophet:** Good for long-term trends and seasonality (yearly cycles).
    """)

st.title("💰 AI Personal Expense Predictor & Advisor")

# ==========================================
# 2. THE ANN MODEL CLASS
# ==========================================
class ANN(nn.Module):
    def __init__(self, input_size=6, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data
def load_and_clean_data():
    file_path = "Personal_Finance_Dataset.csv"
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip() 
    if 'Type' in df.columns:
        df = df[df['Type'] == 'Expense']
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_and_clean_data()

if df is None:
    st.error("❌ File 'Personal_Finance_Dataset.csv' not found.")
    st.stop()

monthly_data = df.groupby(pd.Grouper(key='Date', freq='ME'))['Amount'].sum().reset_index()

# ==========================================
# 4. PREDICTION LOGIC (SWITCHABLE)
# ==========================================
st.divider()
col1, col2 = st.columns(2)

prediction_value = 0.0
model_status_text = ""

# --- LOGIC FOR ANN ---
if model_choice == "ANN (Neural Network)":
    try:
        model = ANN(input_size=6, hidden_size=64)
        state_dict = torch.load("ameer_model.pth", map_location=torch.device('cpu'))
        
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict 
        model.eval()
        
        if len(monthly_data) >= 6:
            last_6_months_raw = monthly_data['Amount'].tail(6).values
            scaler = StandardScaler()
            scaler.fit(monthly_data[['Amount']])
            input_scaled = scaler.transform(last_6_months_raw.reshape(-1, 1))
            input_tensor = torch.tensor(input_scaled.flatten().reshape(1, 6), dtype=torch.float32)
            
            with torch.no_grad():
                pred_scaled = model(input_tensor).item()
                prediction_value = scaler.inverse_transform([[pred_scaled]])[0][0]
                model_status_text = "Powered by PyTorch ANN"
        else:
            st.warning("Not enough data for ANN.")
    except Exception as e:
        st.error(f"ANN Error: {e}")

# --- LOGIC FOR PROPHET ---
elif model_choice == "Facebook Prophet":
    try:
        # Prepare data for Prophet (requires columns 'ds' and 'y')
        prophet_df = monthly_data.rename(columns={'Date': 'ds', 'Amount': 'y'})
        
        # Initialize and Fit (Fast enough to do on-the-fly)
        m = Prophet()
        m.fit(prophet_df)
        
        # Predict 1 month ahead
        future = m.make_future_dataframe(periods=1, freq='M')
        forecast = m.predict(future)
        
        prediction_value = forecast['yhat'].iloc[-1]
        model_status_text = "Powered by Facebook Prophet"
        
    except Exception as e:
        st.error(f"Prophet Error: {e}")

# ==========================================
# 5. DISPLAY RESULTS
# ==========================================

# --- LEFT COLUMN: FORECAST ---
with col1:
    st.markdown("### 🔮 Next Month Forecast")
    if prediction_value > 0:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color: #aaa;">Predicted Total Expense</h4>
            <h1 style="font-size: 3.5rem; color: #feca57; margin: 10px 0;">RM {prediction_value:,.2f}</h1>
            <p style="color: #00d9a5;">{model_status_text}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Waiting for model...")

# --- RIGHT COLUMN: ADVICE & PIE CHART ---
with col2:
    st.markdown("### 🤖 Smart Financial Advice")
    if prediction_value > 0:
        recent_avg = monthly_data['Amount'].iloc[-3:].mean()
        diff = prediction_value - recent_avg
        pct_change = (diff / recent_avg) * 100 if recent_avg != 0 else 0
        
        if pct_change > 5:
            box_class, status = "advice-box-warn", "⚠️ High Spending Alert"
            msg = f"This model predicts spending **{pct_change:.1f}% higher** than your recent average."
        elif pct_change < -5:
            box_class, status = "advice-box-good", "✅ Savings Opportunity"
            msg = f"This model predicts spending **{abs(pct_change):.1f}% lower** than your recent average."
        else:
            box_class, status = "advice-box-good", "⚖️ Stable Budget"
            msg = "Prediction is on track with your recent average."
            
        st.markdown(f"""
        <div class="{box_class}">
            <h4 style="margin-top:0;">{status}</h4>
            <p>{msg}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # PIE CHART (Last 3 Months)
        st.markdown("##### 📂 Recent Spending Habits (Last 3 Months)")
        last_date = df['Date'].max()
        start_date = last_date - pd.DateOffset(months=3)
        recent_data = df[df['Date'] > start_date]
        
        if not recent_data.empty:
            cat_dist = recent_data.groupby('Category')['Amount'].sum().reset_index()
            fig_pie = px.pie(cat_dist, values='Amount', names='Category', hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10),
                                  height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# 6. TREND CHART (UPDATES WITH SELECTION)
# ==========================================
st.divider()
st.subheader("📊 Expense Trend Analysis")

chart_data = monthly_data.copy()

if prediction_value > 0:
    last_date = chart_data['Date'].iloc[-1]
    next_date = last_date + pd.DateOffset(months=1)
    
    new_row = pd.DataFrame({'Date': [next_date], 'Amount': [prediction_value], 'Type': ['Prediction']})
    chart_data['Type'] = 'Actual'
    chart_data = pd.concat([chart_data, new_row], ignore_index=True)
    
    fig = px.line(chart_data, x='Date', y='Amount', color='Type', markers=True,
                  color_discrete_map={'Actual': '#00d9a5', 'Prediction': '#ff4b4b'})
    st.plotly_chart(fig, use_container_width=True)