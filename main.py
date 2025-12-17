import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

#         Step 1: Load the Dataset
df = pd.read_csv('dataset.csv')
#print(df.head())

#          Step 2: Understand the Data
#print(df.shape)
#print(df.dtypes)
#print(df.describe())

#          Step 3: Data Preprocessing
#print(df.isnull())
df["age"] = df["age"].fillna(df["age"].mean())
#print(df.isnull())
#print(df.duplicated())

#        Step 4: Define Input and Output
x = df[["size_sqft","bedrooms","age"]]
y = df["rent"]

#        Step 5: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#        Step 6: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

#       Step 7: Make Predictions
Y_pred = model.predict(X_test)
#print(Y_pred)

#          Step 8: Evaluate the Model
print(mean_absolute_error(y_test, Y_pred))

