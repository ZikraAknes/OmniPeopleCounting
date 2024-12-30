import numpy as np  
import matplotlib.pyplot as plt  

fig = plt.subplots(figsize = (12, 8))
  
X = ['AP','Precision','Recall','F1 Score', 'MSE'] 
Ygirls = [0.819, 0.994, 0.976, 0.985, 0.255] 
Zboys = [0.523, 0.954, 0.982, 0.968, 0.441] 
  
X_axis = np.arange(len(X)) 
  
plt.bar(X_axis - 0.2, Ygirls, 0.38, edgecolor ='white', label = 'YOLO11s') 
plt.bar(X_axis + 0.2, Zboys, 0.38, edgecolor ='white', label = 'SSD MobileNetV2') 

plt.xticks(X_axis, X) 
plt.xlabel("Performance Metrics") 
plt.ylabel("Value") 
plt.title("Training Evaluations") 
plt.legend() 
plt.show() 