from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import numpy as np


mnist = fetch_mldata('MNIST original')
X = mnist['data']
y = mnist['target']

X_train, X_validation_test, y_train, y_validation_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Training set
X_train, y_train = shuffle(X_train, y_train, random_state=0)

X_validation_test, y_validation_test = shuffle(X_validation_test, y_validation_test, random_state=0)

# Validation set
X_validation, y_validation, = X_validation_test[:5000], y_validation_test[:5000]

# Test set
X_test, y_test = X_validation_test[5000:], y_validation_test[5000:]

### Part A
rf_clf1 = RandomForestClassifier(random_state = 0)
rf_clf1.fit(X_train, y_train)
joblib.dump(rf_clf1, "RFClassifier.pkl")

### Part B
ext_clf1 = ExtraTreesClassifier(random_state = 0)
ext_clf1.fit(X_train, y_train)
joblib.dump(ext_clf1, "ETClassifier.pkl")

### Part C
rf_clf2 = RandomForestClassifier(random_state = 0)
ext_clf2 = ExtraTreesClassifier(random_state = 0)
sv_clf = VotingClassifier(estimators=[('rf', rf_clf2), ('et', ext_clf2)], voting='soft')
sv_clf.fit(X_train,y_train)
joblib.dump(sv_clf, "SoftEnsembleClassifier.pkl")

### Part D
def part_d():
    rf_accuracy = accuracy_score(y_test, rf_clf1.predict(X_test))
    ext_accuracy = accuracy_score(y_test, ext_clf1.predict(X_test))
    sv_clf_accuracy = accuracy_score(y_test, sv_clf.predict(X_test))
    return [rf_accuracy, ext_accuracy, sv_clf_accuracy]
joblib.dump(part_d(), "part_d.pkl")


### Part E
merged = np.concatenate((rf_clf1.predict_proba(X_validation), ext_clf1.predict_proba(X_validation)), axis=1)
joblib.dump(merged, "part_e.pkl")

### Part F
rf_clf3 = RandomForestClassifier(random_state = 0)
rf_clf3.fit(merged, y_validation)
joblib.dump(rf_clf3, "Blender.pkl")

### Part G
merged2 = np.concatenate((rf_clf1.predict_proba(X_test), ext_clf1.predict_proba(X_test)), axis=1)
rf_clf3_accuracy = accuracy_score(y_test, rf_clf3.predict(merged2))
joblib.dump(rf_clf3_accuracy, "part_g.pkl")
