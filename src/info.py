def basic_info(df):
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    
    print("\n--- INFO ---")
    df.info()
    
    print("\n--- NUMERICAL ---")
    print(df.describe())
    
    print("\n--- CATEGORICAL ---")
    print(df.describe(include='object'))


def check_missing_data(df):
    print("check value:\n", df.isnull().sum())
    print("percentage of missing:\n", ((df.isnull().sum() / len(df)) * 100).round(2))