import argparse
import joblib
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, hinge_loss
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', type=int, required=True)
parser.add_argument('--C', type=float, required=True)
parser.add_argument('--class_weight', type=int, required=True)
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--kernel', type=str, required=True)
parser.add_argument('--probability', type=int, required=True)
parser.add_argument('--shrinking', type=int, required=True)

args = parser.parse_args()
max_iter = args.max_iter
C = args.C
class_weight = args.class_weight
gamma = args.gamma
kernel = args.kernel
probability = args.probability
shrinking = args.shrinking

df = pd.read_csv("datasets/prepared_data.csv")
labels = list(map(int, df["labels"]))
df = df.drop(['labels'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, random_state=42)

clf = svm.SVC(C=C,  # размер штрафа за неправильную классификацию
              gamma=gamma,
              # параметр, влияющий на значимость одиночно стоящих точек для определения границ
              random_state=42,
              kernel=kernel,  # тип ядра
              # degree=params["degree"], # степень для полиномиального ядра
              shrinking=False if shrinking == 0 else True,  # метод для сокращения вычислений
              max_iter=max_iter,
              class_weight={0: 1, 1: class_weight},
              probability=False if probability == 0 else True)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

joblib.dump(clf, f'models/best_model_{max_iter}_iter.pkl')
print(accuracy)

accuracy = clf.score(X_test.values, y_test)
predicts = clf.predict(X_test.values)
print({"loss": hinge_loss(predicts, y_test), "accuracy": accuracy})
results = pd.DataFrame({"loss": [hinge_loss(predicts, y_test)], "accuracy": [accuracy]})

results.to_csv(f"./metrics/metrics_max_iter_{max_iter}.csv")