import numpy as np
from src.info import check_missing_data



def handle_missing_data(df):
    df = df.copy()

    print("\n==========Before Cleaning===============")
    print(df.head(10))
    check_missing_data(df)

    if df.duplicated().sum():
        df.drop_duplicates(inplace=True)

    df.replace(['', ' ', 'null', 'none', 'na', '--', 'n/a'], np.nan, inplace=True)

    print("\n==========After Replacing===============")
    check_missing_data(df)

    if 'saleprice' in df.columns:
        df['saleprice'] = df['saleprice'].replace(0, np.nan)
        df = df.dropna(subset=['saleprice'])

    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    #numeric_cols = numeric_cols.drop('id', errors='ignore')
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        if df[col].notna().any():
            if abs(df[col].skew()) > 1:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace('nan', np.nan)

        if df[col].notna().sum() > 0:
            unique_ratio = df[col].nunique(dropna=True) / df[col].notna().sum()
            mode_val = df[col].mode()

            if unique_ratio < 0.3 and not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna("unknown")

    print(f"\nProcessed column: {col}")
    check_missing_data(df)

    return df