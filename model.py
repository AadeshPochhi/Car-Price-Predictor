import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


data = pd.read_csv('dataset.csv', na_values='null ')

features = ['Kilometers_Driven_Scaled', 'mileage', 'Age', 'engine_size', 'horsepower', 'Seats', 'Fuel_Type', 'Transmission', 'Owner_Type']

data.dropna(inplace=True)


current_year = 2023

data['Age'] = current_year - data['Year']
data.drop(['Year'], axis=1, inplace=True)

scaler = StandardScaler()
data['Kilometers_Driven_Scaled'] = scaler.fit_transform(data[['Kilometers_Driven']])
joblib.dump(scaler, 'fitted_scaler.pkl')

fuel_type_mapping = {'CNG': 1, 'Diesel': 2, 'Petrol': 3}
data['Fuel_Type'] = data['Fuel_Type'].map(fuel_type_mapping)

data['Transmission'] = (data['Transmission'] == 'Manual').astype(int)

owner_type_mapping = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}
data['Owner_Type'] = data['Owner_Type'].map(owner_type_mapping)


X = data[features]
y = data['Price']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

joblib.dump(model, 'car_price_model.pkl')

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# model2  = LinearRegression()
# model2.fit(X_train, y_train)
# y_pred2 = model2.predict(X_test)
# mse2 = mean_squared_error(y_test, y_pred2)
# print(f'Mean Squared Error (Linear Regression): {mse2}')

# knn_model = KNeighborsRegressor(n_neighbors=5)
# knn_model.fit(X_train, y_train)
# y_pred3 = knn_model.predict(X_test)
# mse3 = mean_squared_error(y_test, y_pred3)
# print(f'Mean Squared Error (K-NN Regression): {mse3}')
