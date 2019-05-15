import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


def draw_elbow(df, n_clusters = list(range(1, 11))):
    """applies preliminary clustering for multiple values of k and produces the elbow plot."""
    clu = df[df.columns.values[:-1]]
    models = [KMeans(n_clusters = i, random_state = 0).fit(clu).inertia_ for i in n_clusters]
    plot = pd.DataFrame(data = {'k' : n_clusters, 'inertia' : models})

    plt.figure(figsize = (15, 6))
    plt.xticks(n_clusters)
    sns.lineplot(data = plot, x = 'k', y = 'inertia')


def cluster(df, n_clusters = 4):
    val = df.iloc[:, :-1].values
    model = KMeans(n_clusters = n_clusters, random_state = 0).fit(val)
    df = df.copy()
    df['cluster'] = model.labels_
    return df


def cluster_stats(df):
    """calculates the percentage of passengers boarding at specific times for each cluster."""
    pattern = list(df.iloc[0])[-2]
    n_days = len(pattern)
    
    cls = [(day + 1, hour) for day in range(n_days) for hour in range(24)]
    tp = pd.DataFrame(columns = cls)
    tp.columns = pd.MultiIndex.from_tuples(tp.columns, names = ['day', 'hour'])
    tp.index.name = 'cluster'
    
    for (key, value) in df.groupby('cluster'):
        d, total = np.zeros((n_days, 24)), len(value)
        for arr in value.iloc[:, :-2].values:
            for i, ax in enumerate(np.split(arr, n_days)):
                ax = np.array([[0, 1][x > 0] for x in ax])
                d[i] += ax
        d /= total
        s = pd.Series({(x + 1, y) : d[x][y] for x in range(n_days) for y in range(24)})
        s.name = key
        tp = tp.append(s)
        
    tp['pattern'] = [pattern] * len(tp)
    return tp