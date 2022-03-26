import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from seaborn import heatmap

# Reading The Data From Excel File Both Test Excel File And Train File
data_set_Train = pd.read_excel("Data_Train (2).xlsx")
data_set_Test = pd.read_excel("Data_Test (2).xlsx")
print(data_set_Train.head())
print(data_set_Test.head())


# Calculating how many are null of each type For Both Test Excel File And Train File
print(data_set_Train.isnull().sum())
print(data_set_Test.isnull().sum())


# Split Function For Data Preparation for both Test Excel File And Train File
def split_data(string):
    string = string.split(" ")
    if string[0] == 'null':
        return None
    return float(string[0])


# Dropping those null data for both Test Excel File And Train File
data_set_Train.dropna(inplace=True)
data_set_Test.dropna(inplace=True)


# Using The Above Split Function for both Test Excel File And Train File
data_set_Train["Mileage"] = data_set_Train["Mileage"].apply(split_data)
data_set_Train["Engine"] = data_set_Train["Engine"].apply(split_data)
data_set_Train["Power"] = data_set_Train["Power"].apply(split_data)

data_set_Test["Mileage"] = data_set_Test["Mileage"].apply(split_data)
data_set_Test["Engine"] = data_set_Test["Engine"].apply(split_data)
data_set_Test["Power"] = data_set_Test["Power"].apply(split_data)


# Dropping those null data for both Test Excel File And Train File
data_set_Train.dropna(inplace=True)
data_set_Test.dropna(inplace=True)

# Calculating how many are null of each type for both Test Excel File And Train File
print(data_set_Train.isnull().sum())
print(data_set_Test.isnull().sum())

# Coping Test Excel File
data_set_Test_ = data_set_Test.copy()

# Seeing the different Columns for both Test Excel File And Train File
print(data_set_Train.columns)
print(data_set_Test.columns)

# Visualization
data_set_Train.hist()
plt.show()
heatmap(data_set_Train.corr())
plt.show()


# Price Vs Kilometers_Driven
plt.scatter(data_set_Train["Price"], data_set_Train["Kilometers_Driven"])
plt.xlabel("Price")
plt.ylabel("Kilometers Driven")
plt.show()

# Price Vs Year
plt.scatter(data_set_Train["Year"], data_set_Train["Price"])
plt.xlabel("Year")
plt.ylabel("Price")
plt.show()


# Encoding Values for both Test Excel File And Train File
# data_set_Train.iloc[:, 5] = LabelEncoder().fit_transform(data_set_Train.iloc[:, 5])
# data_set_Train.iloc[:, 0] = LabelEncoder().fit_transform(data_set_Train.iloc[:, 0])
# data_set_Train.iloc[:, 6] = LabelEncoder().fit_transform(data_set_Train.iloc[:, 6])
# data_set_Train.iloc[:, 4] = LabelEncoder().fit_transform(data_set_Train.iloc[:, 4])
# data_set_Train.iloc[:, 1] = LabelEncoder().fit_transform(data_set_Train.iloc[:, 1])
# data_set_Train.iloc[:, 2] = LabelEncoder().fit_transform(data_set_Train.iloc[:, 2])
#
# data_set_Test.iloc[:, 5] = LabelEncoder().fit_transform(data_set_Test.iloc[:, 5])
# data_set_Test.iloc[:, 0] = LabelEncoder().fit_transform(data_set_Test.iloc[:, 0])
# data_set_Test.iloc[:, 6] = LabelEncoder().fit_transform(data_set_Test.iloc[:, 6])
# data_set_Test.iloc[:, 4] = LabelEncoder().fit_transform(data_set_Test.iloc[:, 4])
# data_set_Test.iloc[:, 1] = LabelEncoder().fit_transform(data_set_Test.iloc[:, 1])
# data_set_Test.iloc[:, 2] = LabelEncoder().fit_transform(data_set_Test.iloc[:, 2])
#
#
# # Extracting Values Training Excel File
# X_train = data_set_Train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
# # Extracting Values Training Excel File
# X_Test_Excel = data_set_Test.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
#
# # Extracting Values Training Excel File
# Y_train = data_set_Train.iloc[:, 11].values
#
# # Adding Columns of 1 Extracting Values Training Excel File
# X_train = np.append(arr=np.ones((len(X_train), 1)).astype(float), values=X_train, axis=1)
# X_Test_Excel = np.append(arr=np.ones((len(X_Test_Excel), 1)).astype(int), values=X_Test_Excel, axis=1)
#
# # Splitting DataSet into Test, Train For Training Excel file
# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, random_state=0, train_size=0.8)
# sc_X = StandardScaler()
# sc_Y = StandardScaler()
# Y_true = Y_test
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
#
#
# print(" ========================================================= ")
# print("Multiple Regression (For Training Excel file)")
# model_1 = LinearRegression().fit(X_train, Y_train)
# predictions = model_1.predict(X_test)
# print(Y_test)
#
# for i in range(len(predictions)):
#     if predictions[i] < 0:
#         predictions[i] = 0
# print(predictions)
# print("Mean Squared Error = ", mean_squared_error(Y_test, predictions))
#
# print(" ========================================================= ")
#
# print("Polynomial Regression (For Training Excel file)")
# polynomial = LinearRegression()
# poly_reg = PolynomialFeatures(degree=2)
# X_poly_train = poly_reg.fit_transform(X_train)
# # poly_reg.fit(X_poly_train, Y_train)
# X_poly_test = poly_reg.transform(X_test)
# model_5 = polynomial.fit(X_poly_train, Y_train)
# predictions = model_5.predict(X_poly_test)
# print(Y_test)
# print(predictions)
# print("Mean Squared Error = ", mean_squared_error(Y_test, predictions))
#
# print(" ========================================================= ")
#
# print("Decision Tree Regression (For Training Excel file)")
# decision_tree = DecisionTreeRegressor(random_state=0)
# model_2 = decision_tree.fit(X_train, Y_train)
# predictions = model_2.predict(X_test)
# print(Y_test)
# print(predictions)
# print("Mean Squared Error = ", mean_squared_error(Y_test, predictions))
#
# print(" ========================================================= ")
#
# print("Random Forest Regression (For Training Excel file)")
# random_forest = RandomForestRegressor(n_estimators=1000)
# model_3 = random_forest.fit(X_train, Y_train)
# predictions = model_3.predict(X_test)
# print(Y_test)
# print(predictions)
# print("Mean Squared Error = ", mean_squared_error(Y_test, predictions))
#
# print(" ========================================================= ")
#
# print("K-Nearest Neighbor (For Training Excel file)")
# k_nearest = KNeighborsRegressor()
# model_4 = k_nearest.fit(X_train, Y_train)
# predictions = model_4.predict(X_test)
# print(Y_test)
# print(predictions)
# print("Mean Squared Error = ", mean_squared_error(Y_test, predictions))
#
# print(" ========================================================= ")
#
# # Test File Excel
# # For Test Set (Only Predictions)
#
# # Scaling X Values
# sc_X = StandardScaler()
# X_Test_Excel = sc_X.fit_transform(X_Test_Excel)
#
# # Predicting Using Multiple Regression
# print("Multiple Regression (For Testing Excel file)")
# predictions_Test = model_1.predict(X_Test_Excel)
# for i in range(len(predictions_Test)):
#     if predictions_Test[i] < 0:
#         predictions_Test[i] = 0
# print(predictions_Test)
# print(" ========================================================= ")
#
# print("Polynomial Regression (For Testing Excel file)")
# X_Test_Excel_poly = poly_reg.transform(X_Test_Excel)
# predictions_Test_1 = model_5.predict(X_Test_Excel_poly)
# for i in range(len(predictions_Test_1)):
#     if predictions_Test_1[i] < 0:
#         predictions_Test_1[i] = 0
# print(predictions_Test_1)
#
# print(" ========================================================= ")
#
# print("Decision Tree (For Testing Excel file)")
# print(model_2.predict(X_Test_Excel))
# print(" ========================================================= ")
#
# print("Random Forest Regression (For Testing Excel file)")
# print(model_3.predict(X_Test_Excel))
# print(" ========================================================= ")
#
# print("K-Nearest Neighbors Regression (For Testing Excel file)")
# print(model_4.predict(X_Test_Excel))
# print(" ========================================================= ")
#
# predictions_1 = predictions_Test
# predictions_2 = model_2.predict(X_Test_Excel)
# predictions_3 = model_3.predict(X_Test_Excel)
# predictions_4 = model_4.predict(X_Test_Excel)
# predictions_5 = predictions_Test_1
#
# data_set_Test_["Prices From Multiple Regression"] = predictions_1
# data_set_Test_["Prices From Decision Tree Regression"] = predictions_2
# data_set_Test_["Prices From Random Fores Regression"] = predictions_3
# data_set_Test_["Prices From K-NN Regression"] = predictions_4
# data_set_Test_["Prices From Polynomial Regression"] = predictions_5
# data_set_Test_.to_csv("Verzeo.csv")
