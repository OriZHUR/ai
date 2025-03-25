import pandas as pd
import plotly.graph_objects as go
df=pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")

print(df)

# בדיקת מספר שורות ועמודות
print(df.shape)
# מספר השורות הוא 53940 ומספר העמודות הוא 10

# בדיקת ערכים חסרים
print(df.isnull().sum())
# מספר הערכים החסרים הוא 0

# הפיכת color  cut clarity למספרים
df['color'] = df['color'].astype('category').cat.codes
df['cut'] = df['cut'].astype('category').cat.codes
df['clarity'] = df['clarity'].astype('category').cat.codes
# נרמאול של כל העמודות
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print(df_scaled)

# יצירת תרשים boxplot לכל העמודות
fig = go.Figure()
[fig.add_trace(go.Box(y=df_scaled[column], name=column)) for column in df_scaled.columns]
fig.show()
# נראה שיש טיפה ערכים חריגים ב xyz בחרתי להשאיר אותם

# יצירת מפת חום של המתאמים עבור כל העמודות
fig = go.Figure()
fig.add_trace(go.Heatmap(z=df_scaled.corr(),
                        x=df_scaled.columns,
                        y=df_scaled.columns))
fig.show()

# 2
# על פי מפת החום carat הכי משפיע 
# יצירת מודל לינארי
X = df['carat']
Y = df['price']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# 3
# יצירת טבלאות עם הנתונים המטרים והמבחנים
train_df = pd.DataFrame({'carat': X_train, 'price': Y_train})
test_df = pd.DataFrame({'carat': X_test, 'price': Y_test})
# 4
# יצירת מודל לינארי
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_df[['carat']], train_df['price'])

# חישוב ניבויים על סט הבדיקה
predictions = model.predict(test_df[['carat']])

# חישוב מדדי ביצוע
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(test_df['price'], predictions)
r2 = r2_score(test_df['price'], predictions)

# הדפסת תוצאות
print("# תוצאות המודל:")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# יצירת גרף השוואה בין ערכים אמיתיים לניבויים
fig = go.Figure()

# הוספת הערכים האמיתיים
fig.add_trace(go.Scatter(x=test_df['carat'], 
                        y=test_df['price'],
                        mode='markers',
                        name='ערכים אמיתיים'))

# הוספת הניבויים
fig.add_trace(go.Scatter(x=test_df['carat'],
                        y=predictions,
                        mode='lines',
                        name='ניבויים'))

fig.update_layout(title='השוואה בין ערכים אמיתיים לניבויים',
                 xaxis_title='Carat',
                 yaxis_title='Price')
fig.show()

# הערכת איכות המודל:
# ה-R2 Score מראה כמה טוב המודל מסביר את השונות בנתונים (בין 0 ל-1, כאשר 1 מושלם)
# מסקנה: על פי ערך ה-R2 שהתקבל, המודל מסביר חלק משמעותי מהשונות במחיר על בסיס משקל היהלום 
# אך עדיין יש מקום לשיפור - ייתכן שכדאי להוסיף משתנים נוספים או להשתמש במודל מורכב יותר


# 5
# יצירת מודל רגרסיה מרובה משתנים
# בחירת המשתנים המובילים
selected_features = ['carat', '', '', '']
X = df[selected_features]


