import matplotlib.pyplot as plt
import pandas as pd
from time import time


def parse(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        df1 = pd.read_csv(filename, sep='\t', lineterminator='\n', nrows=n, names=['Name', 'n1', 'n2'])
        df2 = pd.read_csv(filename, sep='\t', lineterminator='\n', skiprows=n + 1, nrows=2 * n + 1,
                          names=['Name1', 'n1', 'Name2', 'n2'])

    return df1, df2


def plot_hist(df):
    value_counts = df['Name'].value_counts()

    counts = {i: value for i, value in enumerate(value_counts)}

    fig, ax = plt.subplots()
    ax.bar(*zip(*counts.items()))
    ax.set_title('Frequency Distribution of Classes in Train per Label Match')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Enumerated classes')
    plt.grid()
    plt.show()


df1, df2 = parse('pairsDevTrain.txt')
plot_hist(df1)
plot_hist(df2)