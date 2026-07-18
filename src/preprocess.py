import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_and_clean_data():
    # 1. Fetch Data
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df = df.rename(columns={'MedHouseVal': 'HouseValue'})
    
    # 2. Handle Capping Outlier
    df = df[df["HouseValue"] < 5.0]
    
    # 3. Feature Engineering
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]
    
    return df

def get_train_test_splits():
    df = load_and_clean_data()
    
    X = df.drop("HouseValue", axis=1)
    y = df["HouseValue"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_splits()
    print(f"Data Pipeline Ready! Train shape: {X_train.shape}, Test shape: {X_test.shape}")