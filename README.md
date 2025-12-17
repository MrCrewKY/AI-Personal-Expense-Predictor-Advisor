# AI-Personal-Expense-Predictor-Advisor

AI Project Assignment for AI Subject at MJIIT

Group mates:
Ameer (Handle Task 4. and 5.)
Henry (Handle Task 1. to 3.)
Chuah (Handle Task 7. to 8.)

Estimated work flow:
1. Collect datasets on monthly expenses from Hugginface/Kaggle
2. Data Preprocessing (scikit-learn : Clean, normalize, and prepare windows for ML.)
3. Baseline Forecasting (Prophet :Train Prophet to produce baseline time-series predictions.)
4. ANN Training (TensorFlow)
    - Build & train ANN using past 6 months to predict next month.
    - Use MSE loss and Adam optimizer.
    - Plot training/validation loss curves.
5. Forecast Comparison (- Compare Prophet vs ANN using MAE/MSE/MAPE.)
6. Connect ANN to Chatbot MCP
7. Advisory Generation (LLM : Feed forecasts into an LLM to generate budgeting advice.)
8. Dashboard Development (Streamlit : Display historical data, models’ predictions, and AI advice.)
