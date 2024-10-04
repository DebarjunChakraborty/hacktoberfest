# %% [markdown]
# Uber Rides Data Analysis

# %% [markdown]
# Importing all the Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
dataset = pd.read_csv("UberDataset.csv")
dataset.head()

# %%
dataset.shape

# %%
dataset.info()

# %% [markdown]
# Removing Null Values

# %%
dataset['PURPOSE'].fillna("NOT", inplace=True)

# %% [markdown]
# Changing the START_DATE and END_DATE to the date_time format

# %%
dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'], 
									errors='coerce')
dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'], 
									errors='coerce')

# %% [markdown]
# Splitting the START_DATE to date and time column and then converting the time into four different categories i.e. Morning, Afternoon, Evening, Night

# %%
from datetime import datetime
dataset['date'] = pd.DatetimeIndex(dataset['START_DATE']).date
dataset['time'] = pd.DatetimeIndex(dataset['START_DATE']).hour
dataset['day-night'] = pd.cut(x=dataset['time'],
							bins = [0,10,15,19,24],
							labels = ['Morning','Afternoon','Evening','Night'])

# %% [markdown]
# Drop rows with null values

# %%
dataset.dropna(inplace=True)

# %% [markdown]
# Drop the duplicates

# %%
dataset.drop_duplicates(inplace=True)

# %% [markdown]
# Checking the unique values

# %%
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
unique_values = {}
for col in object_cols:
  unique_values[col] = dataset[col].unique().size
unique_values

# %% [markdown]
# Countplot the CATEGORY and PURPOSE

# %%
plt.subplot(1,2,1)
sns.countplot(data=dataset, x='CATEGORY')
plt.xticks(rotation=90)
plt.subplot(1,2,2)
sns.countplot(data=dataset, x='PURPOSE')
plt.xticks(rotation=90)

# %%
sns.countplot(data=dataset, x='day-night')
plt.xticks(rotation=90)
plt.show()

# %%
sns.countplot(data=dataset, x='PURPOSE', hue='CATEGORY')
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# Insights from the above count-plots : 
# 
# 1. Most of the rides are booked for business purpose.
# 2. Most of the people book cabs for Meetings and Meal / Entertain purpose.
# 3. Most of the cabs are booked in the time duration of 10am-5pm (Afternoon).

# %%
from sklearn.preprocessing import OneHotEncoder
object_cols = ['CATEGORY', 'PURPOSE']
dataset.dropna(subset=object_cols, inplace=True)
OH_encoder = OneHotEncoder(sparse=False, drop='first')
OH_cols = OH_encoder.fit_transform(dataset[object_cols])
feature_names = OH_encoder.get_feature_names_out(object_cols)
OH_df = pd.DataFrame(OH_cols, columns=feature_names, index=dataset.index)
df_final = dataset.drop(object_cols, axis=1)
dataset = pd.concat([df_final, OH_df], axis=1)

# %%
numeric_cols = dataset.select_dtypes(include=['number'])
sns.heatmap(numeric_cols.corr(), 
            cmap='BrBG', 
            annot=True, 
            fmt='.2f', 
            linewidths=2)
plt.show()

# %% [markdown]
# Insights from the heatmap:
# 1. Business and Personal Category are highly negatively correlated, this have already proven earlier. So this plot, justifies the above conclusions.
# 
# 2. There is not much correlation between the features.

# %%
dataset['MONTH'] = pd.DatetimeIndex(dataset['START_DATE']).month
month_label = {1.0: 'Jan', 2.0: 'Feb', 3.0: 'Mar', 4.0: 'April',
               5.0: 'May', 6.0: 'June', 7.0: 'July', 8.0: 'Aug',
               9.0: 'Sep', 10.0: 'Oct', 11.0: 'Nov', 12.0: 'Dec'}
dataset["MONTH"] = dataset.MONTH.map(month_label)

monthly_rides_count = dataset['MONTH'].value_counts().sort_index()

max_miles_per_month = dataset.groupby('MONTH')['MILES'].max()

sns.barplot(x=monthly_rides_count.index, y=monthly_rides_count.values)
plt.xlabel("Month")
plt.ylabel("Total Rides Count")
plt.title("Total Rides Count per Month")
plt.xticks(rotation=45)
plt.show()

sns.lineplot(x=max_miles_per_month.index, y=max_miles_per_month.values, marker='o')
plt.xlabel("Month")
plt.ylabel("Maximum Miles Ridden")
plt.title("Maximum Miles Ridden per Month")
plt.xticks(rotation=45)
plt.show()

# %%
dataset['DAY'] = dataset['START_DATE'].dt.weekday
day_label = {
    0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}
dataset['DAY'] = dataset['DAY'].map(day_label)

# %%
day_label = dataset.DAY.value_counts()
sns.barplot(x=day_label.index, y=day_label);
plt.xlabel('DAY')
plt.ylabel('COUNT')

# %%
sns.boxplot(dataset[dataset['MILES']<100]['MILES'])

# %%
sns.distplot(dataset[dataset['MILES']<40]['MILES'])

# %% [markdown]
# Insights from the above plots :
# 1. Most of the cabs booked for the distance of 4-5 miles.
# 2. Majorly people chooses cabs for the distance of 0-20 miles.
# 3. For distance more than 20 miles cab counts is nearly negligible.


