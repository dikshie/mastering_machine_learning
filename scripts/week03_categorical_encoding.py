"""
Week 3: Encoding Categorical Features
Book Chapter: 3 - Encoding categorical features

Topics:
- OneHotEncoder for nominal variables (sparse vs dense output, unknown category handling)
- OrdinalEncoder for ordered categorical features
- LabelEncoder for binary/multiclass target variables
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

def main():
    print("=" * 60)
    print(" Week 3: Categorical Feature Encoding ")
    print("=" * 60)

    # Sample dataset
    df = pd.DataFrame({
        'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'London'],
        'subscribed': ['no', 'yes', 'no', 'yes', 'yes']
    })
    print("\nOriginal DataFrame:")
    print(df)

    # 1. OrdinalEncoder for ordered feature ('education')
    education_order = [['High School', 'Bachelor', 'Master', 'PhD']]
    oe = OrdinalEncoder(categories=education_order)
    df['education_encoded'] = oe.fit_transform(df[['education']])
    print("\n1. Ordinal Encoded ('education'):")
    print(df[['education', 'education_encoded']])

    # 2. OneHotEncoder for nominal feature ('city')
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    city_encoded = ohe.fit_transform(df[['city']])
    city_df = pd.DataFrame(city_encoded, columns=ohe.get_feature_names_out(['city']))
    print("\n2. One-Hot Encoded ('city'):")
    print(city_df)

    # Handling unknown categories in test data
    test_df = pd.DataFrame({'city': ['New York', 'Sydney']}) # 'Sydney' is unknown
    test_encoded = ohe.transform(test_df[['city']])
    print("\n3. Handling Unknown Categories ('Sydney' in test set):")
    print(pd.DataFrame(test_encoded, columns=ohe.get_feature_names_out(['city'])))

    # 4. LabelEncoder for target variable
    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df['subscribed'])
    print("\n4. Label Encoded Target ('subscribed'):")
    print(df[['subscribed', 'target_encoded']])

if __name__ == "__main__":
    main()
