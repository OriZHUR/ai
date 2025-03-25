import pandas as pd
import plotly.graph_objects as go
df = pd.read_csv("https://raw.githubusercontent.com/TzachLectures/AI-lectures---data-files/main/housing_complete_data.csv")
# בדיקה אם יש ערכים חסרים
print(df.isnull().sum())

# בדיקת שונות 0
print(df.nunique())
# הדפסת הטבלה וכל עמודה
print(df.head())

# מחיקת הzipcode
df = df.drop(columns=["zipcode"])
# מחיקת lat וlong
df = df.drop(columns=["lat", "long"])
# מחיקת עמודות שמספר השורות שלהן שונה ממספר השורות של כל הטבלה
df = df.drop(columns=["date"])
# מחיקת id
df = df.drop(columns=["id"])

# כמה חדרים יש בבית עם הכי הרבה חדרים
print(df['bedrooms'].value_counts())
# יצירת עמודה עם הגיל של הבית
from datetime import datetime
curr_year =datetime.now().year
# יצירת עמודה עם הגיל של הבית   
df['age']=curr_year-df['yr_built']
df.drop(columns=['yr_built'], inplace=True)
# יצירת עמודה עם האם הבית נשפך בעבר
import numpy as np
df['ren_was_before']=np.where(df['yr_renovated']==0,0,1)
df.drop(columns=['yr_renovated'], inplace=True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
print(scaled_df)
fig=go.Figure()
[fig.add_trace(go.Box(y=scaled_df[column],name=column)) for column in scaled_df.columns ]
fig.show()
# יצירת מפת חום ולבדוק מה הכי משפיע על המחיר
fig= go.Figure()
fig.add_trace(go.Heatmap(z=scaled_df.corr(),x=scaled_df.columns,y=scaled_df.columns))
# להסיר את העמודה הכי גדולה  בחדרים
scaled_df = scaled_df[scaled_df.bedrooms!=33]
scaled_df.reset_index(drop=True,inplace=True)
# Create a heatmap to visualize correlations between features
fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=df.corr(),
    x=df.columns,
    y=df.columns
))

fig.show()
# להסיר את age ו sqft_above
df.drop(columns=['age','sqft_above'], inplace=True)

# להדפיס את העמודות שיש להם קורונציה מעל 0.5 עם הprice
price_corr=df.corr()['price'][df.corr()['price'].abs()>=0.5]
print(price_corr)

x_df=df[price_corr.index]
y_df=df['price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)


from sklearn.preprocessing import StandardScaler
x_train_scaler = StandardScaler()
x_test_scaler = StandardScaler()
x_train_scaled = x_train_scaler.fit_transform(x_train)
x_test_scaled = x_test_scaler.fit_transform(x_test)


y_train_scaler = StandardScaler()
y_test_scaler = StandardScaler()
y_train_scaled = y_train_scaler.fit_transform(pd.DataFrame(y_train))
y_test_scaled = y_test_scaler.fit_transform(pd.DataFrame(y_test))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train_scaled, y_train_scaled)
intercept = model.intercept_
coef = model.coef_
print(intercept, coef)





