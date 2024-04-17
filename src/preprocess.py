import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from joblib import dump, load

# Define the preprocessing functions
def extract_features_from_name(df):
    # Extracts boolean features based on the presence of keywords in 'Sneaker Name'.
    df['Is_Retro'] = df['Sneaker Name'].str.contains('Retro', case=False, na=False)
    df['Is_Low'] = df['Sneaker Name'].str.contains('Low', case=False, na=False)
    df['Is_High'] = df['Sneaker Name'].str.contains('High', case=False, na=False)
    return df

def create_polynomial_features(df, column_name, degree=2):
    # Creates polynomial features for a given numerical column.
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[[column_name]])
    colnames = [f"{column_name}_poly_{i}" for i in range(1, degree + 1)]
    poly_df = pd.DataFrame(poly_features, columns=colnames, index=df.index)
    df = pd.concat([df, poly_df], axis=1)
    return df

def preprocess_data(df, is_training=True, preprocessor=None):
    # Preprocesses the sneaker dataset.
    if is_training:
        df['Sale Price'] = df['Sale Price'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Retail Price'] = df['Retail Price'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    df['Days Since Release'] = (df['Order Date'] - df['Release Date']).dt.days.fillna(9999)
    df = extract_features_from_name(df)
    df = create_polynomial_features(df, 'Shoe Size')

    if is_training:
        X = df.drop(['Sale Price', 'Order Date', 'Release Date', 'Sneaker Name'], axis=1)
        y = df['Sale Price']
        categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ], remainder='passthrough')
        X_processed = preprocessor.fit_transform(X)
        dump(preprocessor, 'preprocessor.joblib')
        return X_processed, y
    else:
        X = df.drop(['Order Date', 'Release Date', 'Sneaker Name'], axis=1)
        if preprocessor is None:
            preprocessor = load('preprocessor.joblib')
        X_processed = preprocessor.transform(X)
        return X_processed

# If this script is run as the main script, process the entire dataset
if __name__ == "__main__":
    data = pd.read_csv('../datasets/stockx.csv')
    X_processed, y = preprocess_data(data)
    # The variable X_processed can now be used for model training, and y for target values





