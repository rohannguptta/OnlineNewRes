
# Import all dependencies before conducting analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

"""# Dataset 1: Online News Popularity

## Exploratory Data Analysis

For the exploratory data analysis, we will begin by examining the dataset's key features using built-in pandas functions. Additionally, we will check for duplicates or null values, which will be addressed in the data cleaning section.

Finally, we will create several graphs to guide our cleaning decisions. Specifically, we will analyze which day of the week generates the most popular news, identify the categories that tend to be the most popular, and explore the correlation between content length and popularity.
"""

df_news = pd.read_csv("./online_news_popularity.csv")
df_news.head()

df_news.shape

df_news.columns

# Fix column names, remove spaces before name
df_news.columns = df_news.columns.str.strip()

df_news.info()

df_news.describe()

# Check for NULLs
df_news.isnull().sum()

# Check for duplicates
df_news.duplicated().sum()

"""The following graph illustrates the distribution of the 'shares' column. Notably, while most news articles receive relatively few shares, a small number achieve exceptional popularity. During the data cleaning process, we will address these outliers by filtering them out before training."""

sns.histplot(df_news['shares'], bins=50, kde=True)
plt.xlabel('Shares')
plt.title('Distribution of Shares')
plt.show()

# Create column with day of the week
def get_day_of_week(row):
  weekdays = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday',
              'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']
  day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

  for col, day in zip(weekdays, day_names):
      if row[col] == 1:
          return day
  return None

df_news['day_of_week'] = df_news.apply(get_day_of_week, axis=1)

# Calculate the total count and the average number of shares for each day
day_counts = df_news['day_of_week'].value_counts()
average_shares = df_news.groupby('day_of_week')['shares'].mean()

summary = pd.DataFrame({'count': day_counts, 'average_shares': average_shares})
summary = summary.sort_values('average_shares', ascending=False)

summary.head()

"""The following graph illustrates the total number of news articles posted on each day of the week. Additionally, the average number of shares for each day is included to highlight the popularity of news posted on different days. Unsurprisingly, news articles posted on Saturday and Sunday receive the highest average shares. However, these two days also have the fewest number of articles posted. Lastly, it is worth noting that Thursday sees the lowest popularity in terms of news postings."""

plt.figure(figsize=(12, 6))
bars = plt.bar(summary.index, summary['count'], color='gray', edgecolor='black')

for i, (count, avg_share) in enumerate(zip(summary['count'], summary['average_shares'])):
    plt.text(i, count + 10, str(count), ha='center', va='bottom', fontsize=10)
    plt.text(i, count / 2, f"Avg: {avg_share:.0f}", ha='center', va='center', color='black', fontsize=10)

plt.title('Number of News Articles and Average Shares by Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Number of Articles', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

# Create a column that concatenates all data channels for analysis
df_news['data_channel'] = np.select(
    [
        df_news['data_channel_is_lifestyle'] == 1,
        df_news['data_channel_is_entertainment'] == 1,
        df_news['data_channel_is_bus'] == 1,
        df_news['data_channel_is_socmed'] == 1,
        df_news['data_channel_is_tech'] == 1,
        df_news['data_channel_is_world'] == 1,
    ],
    ['lifestyle', 'entertainment', 'bus', 'socmed', 'tech', 'world'],
    default=None
)

df_news['data_channel'].head()

"""The following graph analyzes the popularity by channel. Most categories shared similar popularity. However, we can see Social Media and Lifestyle are not as popular"""

channel_shares = df_news.groupby('data_channel')['shares'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(channel_shares.index, channel_shares.values, color='gray')

for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)

plt.title('Total Shares by Data Channel', fontsize=14)
plt.xlabel('Data Channel', fontsize=12)
plt.ylabel('Total Shares', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()

plt.show()

"""The following scatter plots examine the correlation between several features and popularity, as measured by the number of shares.

Our findings suggest that posts with fewer words tend to perform better, and that news with a negative polarity generally outperforms those with a positive polarity.
"""

plt.figure(figsize=(10, 6))
plt.scatter(df_news['n_tokens_content'], df_news['shares'], alpha=0.5, color='gray')

plt.title('Relationship Between Shares and Number of words', fontsize=14)
plt.xlabel('Number of words', fontsize=12)
plt.ylabel('Number of Shares', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# First plot: Relationship Between Shares and Average Positive Polarity
axs[0].scatter(df_news['avg_positive_polarity'], df_news['shares'], alpha=0.5, color='gray')
axs[0].set_title('Relationship Between Shares and Average Positive Polarity', fontsize=14)
axs[0].set_xlabel('Average Positive Polarity', fontsize=12)
axs[0].set_ylabel('Number of Shares', fontsize=12)
axs[0].grid(True, linestyle='--', alpha=0.7)

# Second plot: Relationship Between Shares and Average Negative Polarity
axs[1].scatter(df_news['avg_negative_polarity'], df_news['shares'], alpha=0.5, color='gray')
axs[1].set_title('Relationship Between Shares and Average Negative Polarity', fontsize=14)
axs[1].set_xlabel('Average Negative Polarity', fontsize=12)
axs[1].set_ylabel('Number of Shares', fontsize=12)
axs[1].grid(True, linestyle='--', alpha=0.7)


plt.tight_layout()
plt.show()

"""## Data Cleaning
In this section, we identify and address outliers in the following columns: **n_tokens_title**, **n_tokens_content**, and **shares**. The goal is to create a dataset with a more uniform distribution, ensuring that our model does not overfit the results.

After the cleaning process was completed, ten thousand records were removed from the dataset. Finally, we can observe that the distributions of these columns are now closer to a normal distribution.
"""

# Title
# Calculate IQR
title_q1 = df_news['n_tokens_title'].quantile(0.25)
title_q3 = df_news['n_tokens_title'].quantile(0.75)
title_iqr = title_q3 - title_q1

# Define Bounds
lower_bound = title_q1 - 1.5 * title_iqr
upper_bound = title_q3 + 1.5 * title_iqr

# Filter outliers
df_news_cleaned = df_news[(df_news['n_tokens_title'] >= lower_bound) & (df_news['n_tokens_title'] <= upper_bound)]

# Content
# Calculate IQR
content_q1 = df_news['n_tokens_content'].quantile(0.25)
content_q3 = df_news['n_tokens_content'].quantile(0.75)
content_iqr = content_q3 - content_q1

# Define Bounds
content_lower = content_q1 - 1.5 * content_iqr
content_upper = content_q3 + 1.5 * content_iqr

# Filter outliers
df_news_cleaned = df_news_cleaned[(df_news_cleaned['n_tokens_content'] >= content_lower) & (df_news_cleaned['n_tokens_content'] <= content_upper)]

# Shares
# Calculate IQR
shares_q1 = df_news['shares'].quantile(0.25)
shares_q3 = df_news['shares'].quantile(0.75)
shares_iqr = shares_q3 - shares_q1

# Define Bounds
shares_lower = shares_q1 - 1.5 * shares_iqr
shares_upper = shares_q3 + 1.5 * shares_iqr

# Filter outliers
df_news_cleaned = df_news_cleaned[(df_news_cleaned['shares'] >= shares_lower) & (df_news_cleaned['shares'] <= shares_upper)]

# Remove Duplicated and NAs
df_news_cleaned = df_news_cleaned.dropna()
df_news_cleaned = df_news_cleaned.drop_duplicates()

print(f"Original dataset shape: {df_news.shape}")
print(f"Dataset shape after removing outliers: {df_news_cleaned.shape}")

sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(df_news_cleaned['n_tokens_title'], kde=True, color='blue', ax=axes[0], stat='density', bins=30)
axes[0].set_title('Distribution of n_tokens_title')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')

sns.histplot(df_news_cleaned['n_tokens_content'], kde=True, color='green', ax=axes[1], stat='density', bins=30)
axes[1].set_title('Distribution of n_tokens_content')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')

sns.histplot(df_news_cleaned['shares'], kde=True, color='red', ax=axes[2], stat='density', bins=30)
axes[2].set_title('Distribution of shares')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Density')

plt.tight_layout()

plt.show()

"""## Dataset Split
For the dataset split, we will use a function from the sklearn package. We will allocate 70% of the data to the training subset and 30% to the test subset for model validation. Additionally, we are setting a seed to ensure the results can be replicated, as several team members are working on this project. This approach guarantees a fair and consistent analysis.
"""

seed = 42

df_train, df_test = train_test_split(df_news_cleaned, test_size=0.3, random_state=seed)

print(f'Training set shape: {df_train.shape}')
print(f'Test set shape: {df_test.shape}')

"""## Training

The models—Support Vector Machine (SVM), k-Nearest Neighbour (KNN), and Decision Tree—will use the same set of features, ensuring a fair comparison. Given that the dataset contains over 60 features, we have selected 19 key features for training. This choice was made to prevent overfitting, as using all the attributes would likely lead to a complex model that fits the training data too closely. Additionally, during our exploratory data analysis, we identified which attributes have the most significant influence, and we have chosen to focus on these for the model.

Since the target variable (shares) is numerical, rather than categorical, we will treat this as a regression problem. Accordingly, we will evaluate the models using the following metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
"""

# Select desired features
# Not using day of the week and data channel as these are very influencial
selected_features = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'num_hrefs',
                     'num_imgs', 'num_videos', 'num_keywords', 'kw_avg_avg',
                     'self_reference_avg_sharess', 'is_weekend', 'LDA_00', 'LDA_01',
                     'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
                     'global_sentiment_polarity', 'avg_positive_polarity', 'avg_negative_polarity']

# Take into account ALL features, but not the strings and target feature
all_features = [col for col in df_train.columns if col != 'shares' and df_train[col].dtype != 'object']

target = 'shares'

X_train_filtered = df_train[selected_features]
y_train_filtered = df_train[target]
X_test_filtered  = df_test[selected_features]
y_test_filtered  = df_test[target]


X_train_all = df_train[all_features]
y_train_all = df_train[target]
X_test_all  = df_test[all_features]
y_test_all  = df_test[target]

# Standardizing features for SVM and KNN
scaler = StandardScaler()

# Filtered
X_train_scaled_all = scaler.fit_transform(X_train_all)
X_test_scaled_all = scaler.transform(X_test_all)

# All
X_train_scaled_filtered = scaler.fit_transform(X_train_filtered)
X_test_scaled_filtered = scaler.transform(X_test_filtered)

"""### SVM"""

# Create Model with ALL features
start_time = time.time()

svm_model_all = SVR(kernel='linear')
svm_model_all.fit(X_train_scaled_all, y_train_all)

end_time = time.time()
svm_training_time_all = end_time - start_time
print(f"Training time: {svm_training_time_all:.2f} seconds")

# Create Model with selected features
start_time = time.time()

svm_model_filtered = SVR(kernel='linear')
svm_model_filtered.fit(X_train_scaled_filtered, y_train_filtered)

end_time = time.time()
svm_training_time_filtered = end_time - start_time
print(f"Training time: {svm_training_time_filtered:.2f} seconds")

# Predictions
start_time = time.time()
svm_y_pred_all = svm_model_all.predict(X_test_scaled_all)
end_time = time.time()
svm_prediction_time_all = end_time - start_time
print(f"Prediction time (All features): {svm_prediction_time_all:.2f} seconds")

start_time = time.time()
svm_y_pred_filtered = svm_model_filtered.predict(X_test_scaled_filtered)
end_time = time.time()
svm_prediction_time_filtered = end_time - start_time
print(f"Prediction time (Selected features): {svm_prediction_time_filtered:.2f} seconds")

# Precision Metrics - ALL
svr_mse_all = mean_squared_error(y_test_all, svm_y_pred_all)
svr_rmse_all = np.sqrt(svr_mse_all)
svr_mae_all = mean_absolute_error(y_test_all, svm_y_pred_all)

svr_results_all = {
    "Model": "SVR (All features)",
    "Training Time": round(svm_training_time_all,3),
    "Prediction Time": round(svm_prediction_time_all,3),
    "Mean Squared Error (MSE)": round(svr_mse_all, 2),
    "Root Mean Squared Error (RMSE)": round(svr_rmse_all, 2),
    "Mean Absolute Error (MAE)": round(svr_mae_all, 2)
}

svr_results_all

# Predictions Metrics - Selected
svr_mse_filtered = mean_squared_error(y_test_filtered, svm_y_pred_filtered)
svr_rmse_filtered = np.sqrt(svr_mse_filtered)
svr_mae_filtered = mean_absolute_error(y_test_all, svm_y_pred_filtered)

svr_results_filtered = {
    "Model": "SVR (Selected features)",
    "Training Time": round(svm_training_time_filtered,3),
    "Prediction Time" : round(svm_prediction_time_filtered,3),
    "Mean Squared Error (MSE)": round(svr_mse_filtered, 2),
    "Root Mean Squared Error (RMSE)": round(svr_rmse_filtered, 2),
    "Mean Absolute Error (MAE)": round(svr_mae_filtered, 2)
}

svr_results_filtered

"""### K-Nearest Neighbour"""

# Create model with ALL features
start_time = time.time()
knn_all = KNeighborsRegressor(n_neighbors=5)
knn_all.fit(X_train_scaled_all, y_train_all)
end_time = time.time()
knn_training_time_all = end_time - start_time
print(f"Training time: {knn_training_time_all:.2f} seconds")

# Create model with selected features
start_time = time.time()
knn_filtered = KNeighborsRegressor(n_neighbors=5)
knn_filtered.fit(X_train_scaled_filtered, y_train_filtered)
end_time = time.time()
knn_training_time_filtered = end_time - start_time
print(f"Training time: {knn_training_time_filtered:.3f} seconds")

# Predictions
start_time = time.time()
knn_y_pred_all = knn_filtered.predict(X_test_scaled_filtered)
end_time = time.time()
knn_prediction_time_all = end_time - start_time
print(f"Prediction time (All features): {knn_prediction_time_all:.2f} seconds")

start_time = time.time()
knn_y_pred_filtered = knn_filtered.predict(X_test_scaled_filtered)
end_time = time.time()
knn_prediction_time_filtered = end_time - start_time
print(f"Prediction time (Selected features): {knn_prediction_time_filtered:.2f} seconds")

# Performance metrics - ALL
knn_mse_all = mean_squared_error(y_test_all, knn_y_pred_all)
knn_rmse_all = np.sqrt(knn_mse_all)
knn_mae_all = mean_absolute_error(y_test_all, knn_y_pred_all)

knn_results_all = {
    "Model": "KNN (All features)",
    "Training Time": round(knn_training_time_all,3),
    "Prediction Time": round(knn_prediction_time_all,3),
    "Mean Squared Error (MSE)": round(knn_mse_all, 2),
    "Root Mean Squared Error (RMSE)": round(knn_rmse_all, 2),
    "Mean Absolute Error (MAE)": round(knn_mae_all, 2)
}

knn_results_all

# Performance metrics - Selected
knn_mse_filtered = mean_squared_error(y_test_filtered, knn_y_pred_filtered)
knn_rmse_filtered = np.sqrt(knn_mse_filtered)
knn_mae_filtered = mean_absolute_error(y_test_filtered, knn_y_pred_filtered)

knn_results_filtered = {
    "Model": "KNN (Selected features)",
    "Training Time": round(knn_training_time_filtered,3),
    "Prediction Time": round(knn_prediction_time_filtered,3),
    "Mean Squared Error (MSE)": round(knn_mse_filtered, 2),
    "Root Mean Squared Error (RMSE)": round(knn_rmse_filtered, 2),
    "Mean Absolute Error (MAE)": round(knn_mae_filtered, 2)
}

knn_results_filtered

"""### Decision Tree"""

start_time = time.time()
dt_model_all = DecisionTreeRegressor(random_state=42)
dt_model_all.fit(X_train_all, y_train_all)
end_time = time.time()
dt_training_time_all = end_time - start_time
print(f"Training time: {dt_training_time_all:.2f} seconds")

start_time = time.time()
dt_model_filtered = DecisionTreeRegressor(random_state=42)
dt_model_filtered.fit(X_train_filtered, y_train_filtered)
end_time = time.time()
dt_training_time_filtered = end_time - start_time
print(f"Training time: {dt_training_time_filtered:.2f} seconds")

# Predictions
start_time =time.time()
dt_y_pred_all = dt_model_all.predict(X_test_all)
end_time = time.time()
dt_prediction_time_all = end_time - start_time
print(f"Prediction time (All features): {dt_prediction_time_all:.3f} seconds")

start_time = time.time()
dt_y_pred_filtered = dt_model_filtered.predict(X_test_filtered)
end_time = time.time()
dt_prediction_time_filtered = end_time - start_time
print(f"Prediction time (Selected features): {dt_prediction_time_filtered:.3f} seconds")

# Performance metrics - ALL
dt_mse_all = mean_squared_error(y_test_all, dt_y_pred_all)
dt_rmse_all = np.sqrt(dt_mse_all)
dt_mae_all = mean_absolute_error(y_test_all, dt_y_pred_all)

dt_results_all = {
    "Model": "Decision Tree (All Features)",
    "Training Time": round(dt_training_time_all,3),
    "Prediction Time": round(dt_prediction_time_all,3),
    "Mean Squared Error (MSE)": round(dt_mse_all, 2),
    "Root Mean Squared Error (RMSE)": round(dt_rmse_all, 2),
    "Mean Absolute Error (MAE)": round(dt_mae_all, 2)
}

dt_results_all

# Performance metrics - Selected
dt_mse_filtered = mean_squared_error(y_test_filtered, dt_y_pred_filtered)
dt_rmse_filtered = np.sqrt(dt_mse_filtered)
dt_mae_filtered = mean_absolute_error(y_test_filtered, dt_y_pred_filtered)

dt_results_filtered = {
    "Model": "Decision Tree (Selected Features)",
    "Training Time": round(dt_training_time_filtered,3),
    "Prediction Time": round(dt_prediction_time_filtered,3),
    "Mean Squared Error (MSE)": round(dt_mse_filtered, 2),
    "Root Mean Squared Error (RMSE)": round(dt_rmse_filtered, 2),
    "Mean Absolute Error (MAE)": round(dt_mae_filtered, 2)
}

dt_results_filtered

"""## Model Comparison"""

df_results = pd.DataFrame([svr_results_all, svr_results_filtered, knn_results_all, knn_results_filtered, dt_results_all, dt_results_filtered])
df_results.head(10)

metrics = ['Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)']
df_long = df_results.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Value')

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df_long, x='Model', y='Value', hue='Metric', palette='viridis')

for bars in ax.containers:
    ax.bar_label(bars, fmt='%.1f', label_type='edge', fontsize=9)

plt.title('Model Comparison by Metrics', fontsize=14)
plt.ylabel('Error Value', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.legend(title='Metric', fontsize=10)
plt.gca().spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

time_metrics = ['Training Time', 'Prediction Time']
df_time = df_results.melt(id_vars='Model', value_vars=time_metrics, var_name='Metric', value_name='Time (seconds)')

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_time, x='Model', y='Time (seconds)', hue='Metric', palette='viridis')

for bars in ax.containers:
    ax.bar_label(bars, fmt='%.2f', label_type='edge', fontsize=9)

plt.title('Training Time vs Prediction Time by Model', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.legend(title='Metric', fontsize=10)
plt.gca().spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_results, x='Model', y='Mean Squared Error (MSE)', hue='Model', palette='viridis')

for bars in ax.containers:
    ax.bar_label(bars, fmt='%.2f', label_type='edge', fontsize=9)

plt.title('Mean Squared Error (MSE) by Model', fontsize=14)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.gca().spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

"""## Dataset 1 Conclusion
While SVM performed the best on the training data, our exploratory data analysis revealed that the data is not linearly separable. The model is likely overfitting to specific nuances of the dataset.
Therefore, we recommend proceeding with the KNN model using only the selected features. This model offers several advantages, including a very short prediction time of 2.474 seconds. Furthermore, KNN produced results that were very similar to those of SVM, but with significantly less training time and without raising concerns about overfitting.

