from scipy.io.arff import loadarff
import pandas as pd
import matplotlib.pyplot as plt

filenames = ['diabetes.arff', 'german_credit.arff']
dfs = {filename.split('.')[0].capitalize(): pd.DataFrame(loadarff(filename)[0]) for filename in filenames}
plt.rcParams["figure.figsize"] = (15, 8)

for filename, df in dfs.items():
    df.hist(bins=50)
    plt.suptitle(filename)
    plt.show()
