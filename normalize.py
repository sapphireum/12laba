import pandas as pd
from sklearn.preprocessing import MinMaxScaler


train = pd.read_csv('data/train.csv', sep=',')
val = pd.read_csv('data/val.csv', sep=',')

scaler = MinMaxScaler()
train = pd.DataFrame(scaler.fit_transform(train))
val = pd.DataFrame(scaler.transform(val))

print(type(train))

train.to_csv('data/train_norm.csv', sep=',', index=False)
val.to_csv('data/val_norm.csv', sep=',', index=False)