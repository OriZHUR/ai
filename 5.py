import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("https://raw.githubusercontent.com/TzachLectures/AI-lectures---data-files/main/cars93.csv")

print(df.head())

# בדיקת nulls
print(df.isnull().sum())
# מחיקת שורות שיש להן nulls
df = df.dropna(axis=0)

print(df.head())
print(df.isnull().sum())

# בדיקת שונות 0
print(df.nunique())

# מחיקת car_name
df = df.drop(columns=["car_name"])

print(df.head())

sns.pairplot(df, hue="origin")
# plt.show()
# להסיר את origin
df = df.drop(columns=["origin"])
# יצירת מפת חום ולבדוק מה הכי משפיע על המחיר
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
print(scaled_df)
fig=go.Figure()
[fig.add_trace(go.Box(y=scaled_df[column],name=column)) for column in scaled_df.columns ]
# fig.show()

fig= go.Figure()
fig.add_trace(go.Heatmap(z=scaled_df.corr(),x=scaled_df.columns,y=scaled_df.columns))
# fig.show()


# לחלק את הנתונים למודל מבחן ומודל תרגול לפי mpg
from sklearn.model_selection import train_test_split
X = scaled_df[["weight"]]
y = scaled_df["mpg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# יצירת מודל ואימון
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# בדיקת המודל
print(model.score(X_test, y_test))
print(model.score(X_train, y_train))

fig,axs = plt.subplots(ncols=4,sharex=True,sharey=True, figsize=(14,5))

sns.regplot(x="weight", y="mpg",data=scaled_df,order=1, ax=axs[0],line_kws={"color":"red"})
sns.regplot(x="weight", y="mpg",data=scaled_df,order=2, ax=axs[1],line_kws={"color":"red"})
sns.regplot(x="weight", y="mpg",data=scaled_df,order=3, ax=axs[2],line_kws={"color":"red"})
sns.regplot(x="weight", y="mpg",data=scaled_df,order=1, ax=axs[3],line_kws={"color":"red"})

plt.show()
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# יצירת מודל ואימון עם נתונים פולינומיאליים
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_train_pred = model.predict(X_train_poly)
# תיקון: יצירת מערך דו-מימדי לפני inverse_transform
y_train_pred_2d = y_train_pred.reshape(-1, 1)
# יצירת מערך מלא בגודל המתאים
full_array = np.zeros((len(y_train_pred), scaled_df.shape[1]))
full_array[:, scaled_df.columns.get_loc("mpg")] = y_train_pred
y_train_pred_rescaled = scaler.inverse_transform(full_array)[:, scaled_df.columns.get_loc("mpg")].reshape(-1,1)

y_train_predicted_rescaled = round(pd.Series(y_train_pred_rescaled[:,0],index=y_train.index),ndigits=2)

train_df = pd.merge(left=X_train, right=y_train, left_index=True, right_index=True)

finally_train_df = pd.merge(left=train_df, right=y_train_predicted_rescaled, left_index=True, right_index=True)

finally_train_df.head()
print(finally_train_df)






