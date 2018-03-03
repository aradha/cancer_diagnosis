from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import mahotas as mh
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def get_feature_vector(example):
    example = np.where(example > 4095, 4095, example).astype('uint16')
    bw = np.where(example > 0, 1, 0)

    vector = []

    #"""
    # Shape Based features
    area = np.sum(example[example.nonzero()])
    vector.append(area)

    eccentricity = mh.features.eccentricity(bw)
    vector.append(eccentricity)

    major, minor = mh.features.ellipse_axes(bw)
    vector.append(major)
    vector.append(minor)

    roundness = mh.features.roundness(bw)
    vector.append(roundness)
    #"""
    # Texture features
    vector += list(mh.features.lbp(example, 4, 8, ignore_zeros=True))

    vector += list(mh.features.pftas(example))

    vector += list(mh.features.zernike_moments(example, 8))

    ## TAKES A LONG TIME TO COMPUTE:
    #vector += list(mh.features.haralick(example,
    #                                    ignore_zeros=True).flatten())
    #vector += list(example.flatten())
    #"""
    return vector

# Input examples are 128 x 128 images
def generate_features(examples):
    features = []
    idx = 0.0
    for example in examples:
        idx += 1

        if idx % 100 == 0:
            print(idx)
        features.append(get_feature_vector(example))

    return np.array(features)

def train_model(model, train_features, train_labels):
    model.fit(train_features, train_labels)
    print("Training Score: ", model.score(train_features, train_labels))
    print("Training Confusion Matrix: \n",
              confusion_matrix(train_labels, model.predict(train_features)))
    return model

def validate_model(model, val_features, val_labels):
    print("Validation Score: ", model.score(val_features, val_labels))
    print("Validation Confusion Matrix: \n",
              confusion_matrix(val_labels, model.predict(val_features)))

def test_model(model, test_features, test_labels):
    print("Test Score: ", model.score(test_features, test_labels))
    print("Test Confusion Matrix: \n",
          confusion_matrix(test_labels, model.predict(test_features)))

def construct_model(train_info, val_info, test_info, has_test_data):
    train_data, train_labels, _ = train_info
    val_data, val_labels, _ = val_info
    if has_test_data:
        test_data, test_labels, _ = test_info

    train_labels = train_labels.argmax(1)
    val_labels = val_labels.argmax(1)
    if has_test_data:
        test_labels = test_labels.argmax(1)

    train_features = generate_features(train_data)
    val_features = generate_features(val_data)
    if has_test_data:
        test_features = generate_features(test_data)

    print("Generated Features")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    if has_test_data:
        test_features = scaler.transform(test_features)

    print("Scaled Data")
    print("Training Model")

    m, n = train_features.shape
    print("Number of Features: ", n)

    # Can switch between models by uncommenting

    #model = SVC(kernel='rbf',
    #            class_weight='balanced')



    #model = LinearSVC(dual=True,
    #                  class_weight='balanced')
    """
    model = LogisticRegression(penalty='l2',
                               dual=False,
                               class_weight='balanced',
                               solver='lbfgs',
                               multi_class='multinomial',
                               max_iter=1000,
                               n_jobs=5)
    """
    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=8,
                                   bootstrap=True,
                                   class_weight='balanced',
                                   n_jobs=8)
    #"""
    model = train_model(model, train_features, train_labels)
    #coeff = model.coef_[0].argsort()[::-1]
    #print(coeff[:6])
    #print(coeff[-5:])
    validate_model(model, val_features, val_labels)
    if has_test_data:
        test_model(model, test_features, test_labels)
