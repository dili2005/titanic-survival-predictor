import pandas as pd

# Load the Titanic dataset directly from the internet
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# First look
print("Shape:", df.shape)
print()
print(df.head())
print()
print("Column info:")
df.info()
print()
print("Missing values:")
print(df.isnull().sum())
print()
print("Survival counts:")
print(df['Survived'].value_counts())