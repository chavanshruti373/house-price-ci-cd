import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    X = data[['area', 'bedrooms', 'location']]
    y = data['price']
    return X, y