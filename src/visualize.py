import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def preprocess(df, col):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    print("Missing after conversion:", df[col].isnull().sum())
    df = df.dropna(subset=[col])
    return df

def plot_hist(df, col):
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=30)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel('Frequency') 
    print("skew:", df[col].skew())
    plt.show() 


def box_plot(df, col):
    plt.boxplot(df[col])
    plt.title(f"{col} Boxplot")
    plt.ylabel(col)
    plt.show()
    return df


def detect_outlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print("lower:", lower, "upper:", upper)

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print("number of outliers:", len(outliers))

    return outliers, lower, upper


def handle_outlier(df, col):
    #df[col] = pd.to_numeric(df[col], errors='coerce')

    outliers, lower, upper = detect_outlier(df, col)
    outlier_ratio = len(outliers) / len(df)

    print("outlier_ratio:", outlier_ratio)

    if outlier_ratio < 0.05:
        print("Few outliers → Removing")
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    elif outlier_ratio < 0.15:
        print("Moderate outliers → Capping")
        df[col] = np.where(df[col] > upper, upper, df[col])
        df[col] = np.where(df[col] < lower, lower, df[col])

    else:
        print("Too many outliers → Log transform")
        df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else x)

    return df

def get_important_features(df):
    corr = df.corr(numeric_only=True)
    top_features = corr['saleprice'].abs().sort_values(ascending=False)
    important_features = top_features[top_features > 0.3].index
    print("Important_Features:\n", important_features)
    df = df[important_features]
    drop_cols = ['garagearea', 'totrmsabvgrd', '1stflrsf']
    df = df.drop(columns=drop_cols, errors='ignore')
    return df

def corr_relation(df):
    plt.figure(figsize=(10,8))
    print("Corr:", df.corr(numeric_only=True))
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.title("Correlation map")
    plt.show()

def scatter_plot(df, col, target='saleprice'):
    plt.scatter(df[col], df[target])
    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(f"{col} vs {target}")
    plt.show()