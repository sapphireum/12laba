import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/25_cc.csv')
train, temp = train_test_split(df, test_size=0.4, random_state=42)
val, test = train_test_split(df, test_size=0.5, random_state=42)

train.to_csv('data/train.csv', index=False)
val.to_csv('data/val.csv', index=False)
test.to_csv('data/test.csv', index=False)