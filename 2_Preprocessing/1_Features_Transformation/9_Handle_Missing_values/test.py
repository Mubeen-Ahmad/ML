import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = np.round(pd.read_csv('Startups.csv')[['R&D Spend','Administration','Marketing Spend','Profit']]/10000)
np.random.seed(9)
df = df.sample(5)
df = df.iloc[:,0:-1]
df.iloc[1,0] = np.NaN
df.iloc[3,1] = np.NaN
df.iloc[-1,-1] = np.NaN



class Mice:
    missed = None
    
    # check arg
    def __init__(self, data, values=None):
        self.data = data
        self.values = values

        if values == None:
            Mice.missed = self.get_missed_value()

        else:
            Mice.missed = self.values

    def get_missed_value(self):
        ind = {}
        for i in self.data.columns:
            r = self.data.loc[self.data[i].isnull()].index
            ind.update({i: list(r)})
        return ind

    def get_mean(self):
        tmp = self.data.copy()
        for i in tmp.columns:
            # add index
            r = self.data.loc[self.data[i].isnull()].index
            tmp[i].fillna(tmp[i].mean(), inplace=True)
        return tmp

    def Algo(self):
        tmp1 = self.data.copy()


        for k, v in Mice.missed.items():
            print(k,v)

            for j in v:
                # replace again nan with column wise
                tmp1[k].loc[j] = np.nan

                # test data
                test = tmp1.loc[j].dropna()

                # split data input and outputs for training
                y = tmp1.dropna()[k]
                x = tmp1.dropna().drop(columns=k).values

                # apply algorithm
                lr = LinearRegression()

                # training
                lr.fit(x, y)

                # pred
                c = lr.predict(test.values.reshape(1, -1))
                print(c)
                c = np.round(c, 1)

                # fill pred (c) value with nan
                tmp1.loc[j].fillna(c[0], inplace=True)

        return tmp1

Main = Mice(df)
new = Main.Algo()
print(new)
