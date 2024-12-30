import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import auc

fields = ['Precision', 'Recall']

df = pd.read_csv('evaluations/reports/ssd_testing.csv', skipinitialspace=True, usecols=fields)

x1 = np.array(df['Recall'].tolist())
y1 = np.array(df['Precision'].tolist())

df = pd.read_csv('evaluations/reports/yolo_testing.csv', skipinitialspace=True, usecols=fields)

x2 = np.array(df['Recall'].tolist())
y2 = np.array(df['Precision'].tolist())

# print('Average Precision (AP): {}'.format(auc(x, y)))

# plt.plot(x, y)
plt.plot(x1, y, color='r', label='SSD MobileNetV2')
plt.plot(x2, y, color='r', label='YOLO11s')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.show()

