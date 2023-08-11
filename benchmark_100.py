import shap
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

number = 100

# Create a bigger initial dataset
df, y = shap.datasets.iris()
df = pd.concat([df] * number, ignore_index=True)
y = np.concatenate([y] * number, axis=0)

# train a SVM classifier
X_train, X_test, Y_train, Y_test = train_test_split(df, y, test_size=0.2, random_state=0)
X_train_summary = shap.kmeans(X_train, 20)

svm = sklearn.svm.SVC(kernel='rbf', probability=True)
svm.fit(X_train, Y_train)

# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(svm.predict_proba, X_train_summary)
shap_values = explainer.shap_values(X_test, nsamples=100)
