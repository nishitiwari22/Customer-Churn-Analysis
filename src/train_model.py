import joblib

import model
from train_model import X_train, y_train

model.fit(X_train, y_train)
joblib.dump(model, "model.joblib")