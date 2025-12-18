# Need to install this when run session every time
!pip install pandas numpy scikit-learn prophet tensorflow matplotlib
import pandas as pd

# Upload csv file in folder named content in Google Colab
df = pd.read_csv("Personal_Finance_Dataset.csv")
df.head()
