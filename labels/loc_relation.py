import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# 示例数据
# X = np.concatenate((np.load('./true_loc_6.npy'), np.load('./true_loc_7.npy'), np.load('./true_loc_8.npy'), np.load('./true_loc_9.npy'), np.load('./true_loc_11.npy'), np.load('./true_loc_12.npy'), np.load('./true_loc_13.npy'), np.load('./true_loc_14.npy'), np.load('./true_loc_15.npy'), np.load('./true_loc_16.npy')), axis=0)   # 输入坐标 (x, y)
# y = np.concatenate((np.load('./avm_loc_6.npy'), np.load('./avm_loc_7.npy'), np.load('./avm_loc_8.npy'), np.load('./avm_loc_9.npy'), np.load('./avm_loc_11.npy'), np.load('./avm_loc_12.npy'), np.load('./avm_loc_13.npy'), np.load('./avm_loc_14.npy'), np.load('./avm_loc_15.npy'), np.load('./avm_loc_16.npy')), axis=0)  # 输出坐标 (x', y')

# X = np.concatenate((np.load('./true_loc_6.npy'), np.load('./true_loc_16.npy')))
# y = np.concatenate((np.load('./avm_loc_6.npy'), np.load('./avm_loc_16.npy')))
suffix = ['0', '20', '40', '60']
avm_data = []
true_data = []
for i in suffix:
    for j in suffix:
        avm_data.append(np.load('./loc2/15/test/avm_loc_16_'+i+'_'+j+'.npy'))
        true_data.append(np.load('./loc2/15/test/true_loc_16_'+i+'_'+j+'.npy'))

X = np.concatenate(true_data, axis=0)
y = np.concatenate(avm_data, axis=0)

# X = np.load('./true_loc_16.npy')
# y = np.load('./avm_loc_16.npy')

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = X, y
X_test = np.load('./loc2/15/test/true_loc_16_0_0.npy')
y_test = np.load('./loc2/15/test/avm_loc_16_0_0.npy')
# X_test = np.load('./loc1/true_loc_16.npy')
# y_test = np.load('./loc1/avm_loc_16.npy')

# # 多项式特征
# degree = 5
# polynomial_features = PolynomialFeatures(degree=degree)

# # 管道
# model = make_pipeline(polynomial_features, LinearRegression())

# # 训练模型
# model.fit(X_train, y_train)

# # 预测
# y_pred = model.predict(X_test)
# print(model.predict(np.array([[70, 1197]])))

# # 评估
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# # 可视化
# plt.scatter(X_test[:, 0], X_test[:, 1], color='blue', label='True values')
# plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted values')

# plt.legend()
# plt.show()

# # 创建随机森林回归模型
# rf_regressor = RandomForestRegressor(n_estimators=1000, random_state=42)

# # 训练模型
# rf_regressor.fit(X_train, y_train)

# # 预测
# y_pred = rf_regressor.predict(X_test)
# print(rf_regressor.predict(np.array([[1044, 1045]])))

# # 评估模型性能
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse:.4f}')
# print(f'R² Score: {r2:.4f}')

# # 可视化
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test[:, 0], X_test[:, 1], color='blue', label='True values')
# plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted values')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Random Forest Regression for Non-linear Relationship')
# plt.legend()
# plt.show()



# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建 MLP 回归模型
mlp_regressor = MLPRegressor(hidden_layer_sizes=(50, 20), activation='relu', solver='adam', max_iter=5000, random_state=42)

# 训练模型
mlp_regressor.fit(X_train_scaled, y_train)

# dump(mlp_regressor, './test.pkl')
# dump(scaler, './test_scaler.pkl')

# 预测
y_pred = mlp_regressor.predict(X_test_scaled)

print(scaler.transform(np.array([[918, 1077]])))
print(mlp_regressor.predict(scaler.transform(np.array([[918, 1077]]))))

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:, 0], y_test[:, 1], color='blue', label='True values')
plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted values')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('MLP Regression for Non-linear Relationship')
plt.legend()
plt.show()