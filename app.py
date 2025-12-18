import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI Expense Predictor Dashboard",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: #e94560;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #e94560, #ff6b6b, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e2a4a, #253a5e);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(233, 69, 96, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .advice-box {
        background: linear-gradient(145deg, #1e3a2f, #254a3e);
        border-radius: 16px;
        padding: 2rem;
        border-left: 4px solid #00d9a5;
        margin: 1rem 0;
    }
    
    .advice-title {
        color: #00d9a5;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .advice-text {
        color: #e0e0e0;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    
    .winner-badge {
        background: linear-gradient(90deg, #00d9a5, #00b894);
        color: #1a1a2e;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin-top: 1rem;
    }
    
    .author-tag {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">💰 AI Expense Predictor & Advisor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Task 8 Dashboard — Comparing Prophet vs ANN Models</p>', unsafe_allow_html=True)

# Placeholder data from notebook
model_data = {
    'Model': ['Prophet', 'ANN'],
    'MAE_Error': [70.00, 12.50],
    'Color': ['#e94560', '#00d9a5']
}

df_models = pd.DataFrame(model_data)

# Layout columns
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📊 Model Error Comparison (MAE)")
    
    # Create bar chart with Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_models['Model'],
        y=df_models['MAE_Error'],
        marker=dict(
            color=['#e94560', '#00d9a5'],
            line=dict(color='rgba(255,255,255,0.3)', width=2)
        ),
        text=[f'${x:.2f}' for x in df_models['MAE_Error']],
        textposition='outside',
        textfont=dict(size=18, color='white', family='Outfit'),
        hovertemplate='<b>%{x}</b><br>Error: $%{y:.2f}<extra></extra>'
    ))
    
    # --- FIX APPLIED BELOW ---
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Outfit'),
        yaxis=dict(
            title=dict(
                text='Mean Absolute Error ($)',
                font=dict(size=14)
            ),
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 90]
        ),
        xaxis=dict(
            title=dict(text=""),
            tickfont=dict(size=16)
        ),
        height=400,
        margin=dict(t=40, b=40, l=60, r=40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Winner announcement
    st.markdown("""
    <div style="text-align: center; margin-top: -1rem;">
        <span class="winner-badge">🏆 ANN Model Wins — 82% Lower Error!</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🤖 AI Financial Advice")
    
    # Spending trend indicator
    spending_trend = -0.6
    trend_color = "#00d9a5" if spending_trend < 0 else "#e94560"
    trend_icon = "📉" if spending_trend < 0 else "📈"
    
    st.markdown(f"""
    <div class="advice-box">
        <div class="advice-title">{trend_icon} Spending Trend: {spending_trend}%</div>
        <div class="advice-text">
            Based on the ANN model's prediction, your spending is forecasted to 
            <strong style="color: {trend_color};">decrease slightly</strong> next month. 
            The biggest spending category is <strong style="color: #feca57;">Shopping</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # AI-generated tips
    st.markdown("#### 💡 Personalized Tips")
    
    tips = [
        ("🛒", "Review Shopping Habits", "Consider setting a monthly shopping budget limit. Track impulse purchases and wait 24 hours before big buys."),
        ("📅", "Plan Ahead", "Create a weekly meal plan to reduce food expenses. Batch cooking can save both time and money."),
        ("💳", "Monitor Subscriptions", "Audit recurring payments quarterly. Cancel unused services and negotiate better rates where possible.")
    ]
    
    for icon, title, desc in tips:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border-left: 3px solid #feca57;">
            <div style="font-size: 1.1rem; font-weight: 600; color: #feca57;">{icon} {title}</div>
            <div style="color: #b0b0b0; font-size: 0.95rem; margin-top: 0.3rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Divider
st.markdown("<hr style='border: 1px solid #333; margin: 2rem 0;'>", unsafe_allow_html=True)

# Summary metrics row
st.markdown("### 📈 Quick Summary")
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.metric(label="Prophet MAE", value="$70.00", delta="-57.50 vs ANN", delta_color="inverse")

with col_b:
    st.metric(label="ANN MAE", value="$12.50", delta="Best Model", delta_color="normal")

with col_c:
    st.metric(label="Spending Trend", value="-0.6%", delta="Decreasing", delta_color="normal")

with col_d:
    st.metric(label="Top Category", value="Shopping", delta="Monitor closely")

# Footer
st.markdown("""
<div class="author-tag">
    Built by <strong>Chuah</strong> • AI Personal Expense Predictor & Advisor Project • Task 8 Dashboard
</div>
""", unsafe_allow_html=True)

