import pandas as pd
from sklearn.utils import shuffle
real = pd.read_csv("./input/True.csv")
fake = pd.read_csv("./input/Fake.csv")

num = [1000, 5000, 10000, 15000, 20000, 30000, 40000]
i = 0
for k in num:
    data = pd.concat([real, fake], axis=0)
    data = shuffle(data)
    data = data[:k]
    name = './input/Data_test_' + str(i) + '.csv'
    data.to_csv(name, index=False)
    i = i + 1
