import src.preprocessing as p
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(15, 5)})
plt.figure(figsize=(20,20))
sns.heatmap(data=p.train_select.corr(),square=True,cmap="Blues",annot=True)
plt.show()