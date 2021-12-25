import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import itertools
import time


def classif(x, y):
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True,stratify=y)
    std = StandardScaler()
    x = std.fit_transform(x)
    knn = KNN(n_neighbors=2)
    score = cross_val_score(knn, x, y, cv=3, scoring='f1')
    # knn.fit(X_train,y_train)
    # pred = knn.predict(X_test)
    return score.mean()


File = open('Log.txt', 'w')
data = pd.read_csv('prostate_preprocessed.txt', sep=" ", header=None)
data = data.T
col = data.iloc[0]
data.columns = col
data = data.drop([0], axis=0)
y = pd.factorize(data.iloc[:, -1])[0]
columns = list(data.columns[:-1].values)
candid = []
best = []
c = 1
while columns:
    print(c)
    start = time.time()
    if len(candid) == 0:
        p = list(itertools.permutations(columns, 1))
    else:
        p = list(tuple(candid + [x]) for x in columns)
    temp = dict((key, 0) for key in p)
    for per in p:
        x = data.iloc[:][list(per)].values
        temp[per] = classif(x, y)
    b = max(temp, key=temp.get)
    best.append({'Columns': b, 'Score': temp[b]})
    end = time.time()
    t = end - start
    print(c, t, b, temp[b])
    print(c, t, b, temp[b], file=File)
    if len(candid) == 0:
        columns.remove(b[0])
        candid.append(b[0])
    else:
        candid = [x for x in b]
        columns.remove(b[-1])
    c += 1
best = sorted(best, key=lambda x: x['Score'], reverse=True)
print('Best Subsequenc :', best[0], File)
File.close()
