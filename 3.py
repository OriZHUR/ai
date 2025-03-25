import pandas as pd
import plotly.graph_objects as go
df = pd.read_csv("https://raw.githubusercontent.com/TzachLectures/AI-lectures---data-files/main/HousingPrices.csv")
# print the first 5 rows of the dataframe
print(df.head())
# print the last 5 rows of the dataframe
print(df.tail())
# print the info of the dataframe

print(df.info())

# בדיקה אם יש עמודות עם שונות 0
print(df.nunique())
# נרמאול כל העמודות בנרמאול min max
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
print(scaled_df)

# מציאת outliers בהכל העמודות
fig=go.Figure()
[fig.add_trace(go.Box(y=scaled_df[column],name=column)) for column in scaled_df.columns ]
fig.show()
# יצירת מפת חום ולבדוק מה הכי משפיע על המחיר
fig= go.Figure()
fig.add_trace(go.Heatmap(z=scaled_df.corr(),x=scaled_df.columns,y=scaled_df.columns))
fig.show()
# יצירת מודל לינארי
X = scaled_df['LivingArea']
Y = scaled_df['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# יצירת טבלאות עם הנתונים המטרים והמבחנים
train_df = pd.DataFrame({'LivingArea': X_train, 'Price': Y_train})
test_df = pd.DataFrame({'LivingArea': X_test, 'Price': Y_test})
# יצירת מודל לינארי
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_df[['LivingArea']], train_df['Price'])
# מציאת השיפוע והחיתוך של המודל
slope = model.coef_[0]
intercept = model.intercept_
print(f"slope: {slope}, intercept: {intercept}")
predictions = model.predict(train_df[['LivingArea']])
print(predictions)
# ביטול הנרצמאול של המודל
predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))
print(predicted_prices)


