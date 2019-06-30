from PIL import Image
import numpy as np
import glob
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

### Preprocessing
X_train = []
y_train_directionfaced = []
X_test = []
y_test_directionfaced = []

def image_to_traindata(image_path, image_name):
    image_array = np.array((Image.open(image_path+image_name)).convert("L"))
    image_vector = image_array.flatten()
    direction_encode = {'right': 0, 'left': 1, 'up': 2, 'straight': 3}
    direction_face = image_name.split("_")[1]
    direction_face_encoded = direction_encode[direction_face]
    return image_vector, direction_face_encoded

training_set = glob.glob("TrainingSet/*.jpg")
for i in training_set:
    image_path = i.split("\\")[0] + "/"
    image_name = i.split("\\")[1]
    image_vector, direction_face_encoded = image_to_traindata(image_path,image_name)
    X_train.append(image_vector)
    y_train_directionfaced.append(direction_face_encoded)
X_train = np.array(X_train)
y_train_directionfaced = np.array(y_train_directionfaced)

test_set = glob.glob("TestSet/*.jpg")
for i in test_set:
    image_path = i.split("\\")[0] + "/"
    image_name = i.split("\\")[1]
    image_vector, direction_face_encoded = image_to_traindata(image_path,image_name)
    X_test.append(image_vector)
    y_test_directionfaced.append(direction_face_encoded)
X_test = np.array(X_test)
y_test_directionfaced = np.array(y_test_directionfaced)

########################################################################################################################

### part a
def part_a():
    rf_clf = RandomForestClassifier(random_state = 0)
    start_time = time.time()
    rf_clf.fit(X_train, y_train_directionfaced)
    execution_time = time.time() - start_time
    rf_clf_accuracy = accuracy_score(y_test_directionfaced, rf_clf.predict(X_test))
    return rf_clf, execution_time, rf_clf_accuracy
rf_clf, execution_time, rf_clf_accuracy = part_a()
joblib.dump([rf_clf,execution_time,rf_clf_accuracy], "part_a.pkl", protocol=2)

########################################################################################################################

### part b
def part_b():
    pca = PCA(n_components=0.95)
    train_features = pca.fit_transform(X_train)
    rf_clf = RandomForestClassifier(random_state = 0)
    start_time = time.time()
    rf_clf.fit(train_features, y_train_directionfaced)
    execution_time = time.time() - start_time
    test_features = pca.transform(X_test)
    rf_clf_accuracy = accuracy_score(y_test_directionfaced, rf_clf.predict(test_features))
    return rf_clf, execution_time, rf_clf_accuracy
rf_clf, execution_time, rf_clf_accuracy = part_b()
joblib.dump([rf_clf,execution_time,rf_clf_accuracy], "part_b.pkl", protocol=2)

########################################################################################################################

### preprocessing of part c and part d for emotions
y_train_emotion = []
y_test_emotion = []

def image_to_traindata_emotion(image_path, image_name):
    emotion_encode = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3}
    emotion = image_name.split("_")[2]
    emotion_face_encoded = emotion_encode[emotion]
    return emotion_face_encoded

training_set = glob.glob("TrainingSet/*.jpg")
for i in training_set:
    image_path = i.split("\\")[0] + "/"
    image_name = i.split("\\")[1]
    emotion_face_encoded = image_to_traindata_emotion(image_path, image_name)
    y_train_emotion.append(emotion_face_encoded)
y_train_emotion = np.array(y_train_emotion)

test_set = glob.glob("TestSet/*.jpg")
for i in test_set:
    image_path = i.split("\\")[0] + "/"
    image_name = i.split("\\")[1]
    emotion_face_encoded = image_to_traindata_emotion(image_path, image_name)
    y_test_emotion.append(emotion_face_encoded)
y_test_emotion = np.array(y_test_emotion)



########################################################################################################################

### part c
def part_c():
    lr_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=0)
    start_time = time.time()
    lr_clf.fit(X_train, y_train_emotion)
    execution_time = time.time() - start_time
    lr_clf_accuracy = accuracy_score(y_test_emotion, lr_clf.predict(X_test))
    return lr_clf, execution_time, lr_clf_accuracy
lr_clf, execution_time, lr_clf_accuracy = part_c()
joblib.dump([lr_clf, execution_time, lr_clf_accuracy], "part_c.pkl", protocol=2)

########################################################################################################################

### part d
def part_d():
    pca = PCA(n_components=0.95)
    train_features = pca.fit_transform(X_train)
    lr_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=0)
    start_time = time.time()
    lr_clf.fit(train_features, y_train_emotion)
    execution_time = time.time() - start_time
    test_features = pca.transform(X_test)
    lr_clf_accuracy = accuracy_score(y_test_emotion, lr_clf.predict(test_features))
    return lr_clf, execution_time, lr_clf_accuracy
lr_clf, execution_time, lr_clf_accuracy = part_d()
joblib.dump([lr_clf, execution_time, lr_clf_accuracy], "part_d.pkl", protocol=2)

########################################################################################################################