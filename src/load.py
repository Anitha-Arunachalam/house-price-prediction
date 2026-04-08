import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully")
        df.columns = df.columns.str.strip().str.lower()
        return df
    except FileNotFoundError:
        print("file Not Found")
        return None