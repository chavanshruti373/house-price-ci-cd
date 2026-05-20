from sklearn.metrics import r2_score
from src.train import train_model

def evaluate_model():
    model, X_test, y_test = train_model()

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    print(f"R2 Score: {score}")

    return score

if __name__ == "__main__":
    evaluate_model()