from src.data_preprocessing import load_data

def test_data_loading():
    data = load_data("data/raw/house_data.csv")

    assert data is not None
    assert len(data) > 0