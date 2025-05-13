import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


suffix = ['0', '20', '40', '60']
avm_data = []
true_data = []
for i in suffix:
    for j in suffix:
        if i == 0 and j == 0:
            continue
        avm_data.append(np.load('./loc2/15/test/avm_loc_16_'+i+'_'+j+'.npy'))
        true_data.append(np.load('./loc2/15/test/true_loc_16_'+i+'_'+j+'.npy'))

X_train = np.concatenate(true_data, axis=0)
y_train = np.concatenate(avm_data, axis=0)

# 划分训练集和测试集
X_test = np.load('./loc2/15/test/true_loc_16_0_0.npy')
y_test = np.load('./loc2/15/test/avm_loc_16_0_0.npy')
# X_test = np.load('./loc1/true_loc_16.npy')
# y_test = np.load('./loc1/avm_loc_16.npy')

# 二维插值示例
# 为了进行二维插值，我们需要将输入数据和输出数据组合起来
points = X_train
values = y_train

np.save('points.npy', points)
np.save('values.npy', values)

# 评估插值结果
y_test_pred = griddata(points, values, X_test, method='linear')
mse = np.mean((y_test - y_test_pred) ** 2)
r2 = 1 - (np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:, 0], y_test[:, 1], color='blue', label='True values')
plt.scatter(y_test_pred[:, 0], y_test_pred[:, 1], color='red', label='Predicted values')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Griddata for Non-linear Relationship')
plt.legend()
plt.show()