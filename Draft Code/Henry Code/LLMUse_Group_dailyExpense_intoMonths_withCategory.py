category_monthly = (
    df_expense
    .groupby([
        pd.Grouper(key='Date', freq='ME'),
        'Category'
    ])['Amount']
    .sum()
    .reset_index()
)

category_monthly.columns = ['date', 'category', 'expense']
category_monthly.head()
