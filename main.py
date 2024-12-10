import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Loading the dataset
file_path = "data/wdbc-converted.csv"
df = pd.read_csv(file_path)

# Selecting the columns I want to use in the dataset (for display only)
columns_to_display = ['id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'Cancer_Type']

# Splitting the data into training and testing
train_df, test_df, = train_test_split(df, train_size=360, random_state=42) # Training 360 samples | Testing 209 samples

print("\n============================================================================\n")
print("\t\tEARLY-STAGE BREAST CANCER DETECTION MODEL")
print("\n============================================================================\n")

# Verifying the split worked
print("---------------------------------------")
print(f"\tTraining set size: {len(train_df)}")
print(f"\tTesting set size: {len(test_df)}")
print("---------------------------------------\n")

# Building and determining 'target column'
target_column = 'diagnosis'
X_train = train_df.drop(columns=[target_column, 'id'])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column, 'id'])
y_test = test_df[target_column]

# Normalizing data (if the features vary significantly, we need to normalize them)
scaler = StandardScaler()

# Fit and transform the training data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initializing and training the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Making the prediction
y_pred = model.predict(X_test)

# The models accuracy score
print("---------------------------------------")
accuracy = accuracy_score(y_test, y_pred)
print(f"\tModel Accuracy: {accuracy * 100:.2f}%") # Percentage instead of a decimal
print("---------------------------------------\n")

# Persicion, F1 Score, etc.
print("\t\tClassification Report")
print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- ")
print(classification_report(y_test, y_pred))
print("-------------------------------------------------------\n")

# Displaying the results
test_df['Prediction'] = y_pred

# Decided to add a column that displays the diagnosis
test_df['Cancer_Type'] = test_df['Prediction'].apply(lambda x: 'Benign (B)' if x == 'B' else 'Malignant (M)')

# Renamed columns for better readability
display_columns = {
    'id': 'ID',
    'radius_mean': 'Radius (Mean)',
    'texture_mean':  'Texture (Mean)',
    'perimeter_mean':  'Perimeter (Mean)',
    'area_mean':  'Area (Mean)',
    'smoothness_mean':  'Smoothness (Mean)',
    'Cancer_Type':  'Diagnosis (B/M)'
}

print("\t\t\t\t\t\tModel Results")
print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -")
print(test_df.rename(columns=display_columns)[list(display_columns.values())])
print("=============================================================================================================")