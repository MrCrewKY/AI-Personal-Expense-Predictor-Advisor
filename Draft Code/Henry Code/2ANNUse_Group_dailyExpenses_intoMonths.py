df_expense = df[df['Type'] == 'Expense']
df_expense['Date'] = pd.to_datetime(df_expense['Date'])
monthly_expense = (
    df_expense
    .groupby(pd.Grouper(key='Date', freq='ME'))['Amount']
    .sum()
    .reset_index()
)

monthly_expense.columns = ['date', 'expense']
monthly_expense.head()
