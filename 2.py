import pandas as pd

weather_df = pd.read_csv('https://raw.githubusercontent.com/TzachLectures/AI-lectures---data-files/main/weather_data.csv', parse_dates=['day'], index_col='day')

weather_df.rename(columns={'temperature': 'temp', 'windspeed': 'wind_speed', 'event': 'event'}, inplace=True)
print(weather_df)
# בדיקה כמה שורות בעל ערכים רקיים
weather_df.isnull().any().sum()
print(weather_df.isnull().any().sum())
# מילוי העמודות wind_speed בערכים רקיים בעזרת הממוצע
weather_df['wind_speed'].fillna(weather_df['wind_speed'].mean(), inplace=True)
# print(weather_df)
# מחיקת הערכים הרקים בevent 
weather_df.dropna(subset=['event'], inplace=True)
print(weather_df)

# שינוי תקין למספרים
weather_df['event'].unique()
weather_df['event'].replace({'Sunny': 0, 'Cloudy': 1, 'Rain': 2, 'Snow': 3}, inplace=True)
print(weather_df)
# מחיקת העמודה day_type
weather_df.drop(columns=['day_type'], inplace=True)

# print(weather_df)

# בtemp במה שחסר תשלים עם חציון
# weather_df['temp'].fillna(weather_df['temp'].median(), inplace=True)
# print(weather_df)

# מילוי ערכים חסרים בעמודת temp בעזרת שיטת האינטרפולציה
weather_df['temp']=weather_df.temp.interpolate()
print(weather_df)

import plotly.graph_objects as go
fig=go.Figure()
fig.add_trace(go.Heatmap(
    z=weather_df.corr()
    ,x=weather_df.columns,
    y=weather_df.columns))
fig.show()