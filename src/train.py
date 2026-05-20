from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from src.data_preprocessing import load_data, preprocess_data

def train_model():
    data = load_data("data/raw/house_data.csv")

    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

    return model, X_test, y_test

if __name__ == "__main__":
    train_model()