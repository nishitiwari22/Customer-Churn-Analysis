import pandas as pd

def make_prediction(model, input_data, feature_order):
    """
    model: trained model (or pipeline)
    input_data: dict of user inputs
    feature_order: list of features used during training
    """

    # Convert dict → DataFrame with correct order
    data = pd.DataFrame([input_data])[feature_order]

    # Predict
    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

    return prediction[0], probability