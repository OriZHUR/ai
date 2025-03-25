import pandas as pd

heart_df = pd.read_csv("https://raw.githubusercontent.com/UrielBender/BigData/master/DataSets/heart.csv")

heart_df.columns = ['first_name', 'age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'a1c_test', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'liver_function', 'tilt_test','target']

head = heart_df.head()
print(head)

columns = heart_df.columns
print(columns)

info = heart_df.info()
print(info)

describe = heart_df.describe()
print(describe)

# בדיקת כפילויות
duplicates = heart_df.duplicated()
print("number of duplicates:", duplicates.sum())

# הצגת השורות הכפולות
duplicate_rows = heart_df[duplicates]
print("duplicate rows:")
print(duplicate_rows)

# מחיקת שורות כפולות
heart_df = heart_df.drop_duplicates()
print("number of rows after removing duplicates:", len(heart_df))

# הדפסת הטבלה לאחר המחיקה
print(heart_df)
# התחול הindex
heart_df.reset_index(drop=True, inplace=True)
print(heart_df)

# בדיקת שורות ריקות
empty_rows = heart_df[heart_df.isnull()]
print("number of empty rows:", empty_rows.shape[0])

# אם חסר בtilt_test אז נמלא ב0
heart_df['tilt_test'] = heart_df['tilt_test'].fillna(0)
# הדפסת הטבלה לאחר המלאה
print(heart_df)

# לכניס לחלקים הרקים בchoeleseter את הממוצע של העמודה
mean_cholesterol = heart_df['cholesterol'].mean()
heart_df['cholesterol'] = heart_df['cholesterol'].fillna(mean_cholesterol)
print(heart_df)    

# להכניס בעמודת ה sex את המין השכיח
most_common_sex = heart_df['sex'].mode()[0]
heart_df['sex'] = heart_df['sex'].fillna(most_common_sex)
print(heart_df)

# להכניס בmax_heart_rate_achieved חציון של העמודה
median_max_heart_rate = heart_df['max_heart_rate_achieved'].median()
heart_df['max_heart_rate_achieved'] = heart_df['max_heart_rate_achieved'].fillna(median_max_heart_rate)
print(heart_df)

# בדיקת מין בעמודת ה sex
heart_df["sex"].unique()

# בדיקת מין בעמודת ה sex
set(heart_df["sex"])

# שינוי מין למספרים
heart_df.replace({"sex": {"male": 1, "female": 0}}, inplace=True)

# שינוי תקין למספרים
heart_df['target'] = heart_df['target'].map({'yes': 1, 'no': 0})
# הדפסה
print(heart_df)

heart_df.drop(columns=['first_name'], inplace=True)

# בדיקת שונות 0
# print(heart_df['target'].value_counts())

# תרשים של הקשר בין העמודות
import plotly.graph_objects as go
fig=go.Figure()
fig.add_trace(go.Heatmap(
    z=heart_df.corr()
    ,x=heart_df.columns,
    y=heart_df.columns))
# fig.show();          

# מחיקת העמודה a1c_test
heart_df[['a1c_test','fasting_blood_sugar']]
heart_df.drop(columns=['a1c_test'],inplace=True)

# תרשים boxplot על העמודות 
fig=go.Figure()
fig.add_trace(go.Box(y=heart_df['fasting_blood_sugar'], marker=dict(outliercolor='red'), name='fasting_blood_sugar'))
# fig.show();

# פונקציה שמחזירה את הערכים שהם מחוץ לטווח הממוצע והסטיית התקן
def get_outliers(column_name):
    q1 = heart_df[column_name].quantile(0.25)
    q3 = heart_df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return heart_df[(heart_df[column_name] < lower_bound) | (heart_df[column_name] > upper_bound)]

print(get_outliers('fasting_blood_sugar'))

# להסיר את הערכים שהם מחוץ לטווח הממוצע והסטיית התקן
lower_bound = heart_df['fasting_blood_sugar'].quantile(0.25)
upper_bound = heart_df['fasting_blood_sugar'].quantile(0.75)
heart_df = heart_df[(heart_df['fasting_blood_sugar'] >= lower_bound) & (heart_df['fasting_blood_sugar'] <= upper_bound)]
print(heart_df)
 

# # נרמול של העמודות בz score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
heart_df[['max_heart_rate_achieved']] = scaler.fit_transform(heart_df[['max_heart_rate_achieved']])
print(heart_df)
# לעשות ממוצע של נתונים מנורמלים
mean_max_heart_rate = heart_df['max_heart_rate_achieved'].mean()
print(mean_max_heart_rate)

#  נרמול של העמודות בmin max
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
heart_df[['cholesterol']] = scaler.fit_transform(heart_df[['cholesterol']])
print(heart_df)

