import pandas as pd
import optuna
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hinge_loss

df = pd.read_csv("../datasets/prepared_data.csv")

labels = list(map(int, df["labels"]))
df = df.drop(['labels'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, random_state=42)

print(type(X_test))
print(X_test.values)
kernel =['linear', 'poly', 'rbf', 'sigmoid']

def objective(trial):
    params = {
        "C": trial.suggest_loguniform('C', 1e-5, 100),
        "gamma": trial.suggest_loguniform('gamma', 1e-5, 10),
        "kernel": trial.suggest_categorical("kernel", kernel),
        #"degree": trial.suggest_int("degree", low=3, high=5),
        "shrinking": trial.suggest_int("shrinking", low=0, high=1),
        "max_df": trial.suggest_float("max_df", 0.75, 0.9),
        "max_iter": trial.suggest_int("max_iter", low=20, high=1000, step=10),
        "class_weight": trial.suggest_int("class_weight", low=2, high=8),
        "probability": trial.suggest_int("probability", low=0, high=1),
        "accuracy": 0

    }
    print(params)

    clf = svm.SVC(C=params["C"], # размер штрафа за неправильную классификацию
                  gamma=params["gamma"], # параметр, влияющий на значимость одиночно стоящих точек для определения границ
                  random_state=42,
                  kernel=params["kernel"], # тип ядра
                  #degree=params["degree"], # степень для полиномиального ядра
                  shrinking=False if params["shrinking"] == 0 else True, # метод для сокращения вычислений
                  max_iter=params["max_iter"],
                  class_weight={0: 1, 1: params["class_weight"]},
                  probability=False if params["probability"] == 0 else True
                  )

    clf.fit(X_train.values, y_train)
    accuracy = clf.score(X_test.values, y_test)

    predicts = clf.predict(X_test.values)
    loss = hinge_loss(predicts, y_test)
    print(loss)
    print(classification_report(predicts, y_test))
    print(accuracy)
    params["accuracy"] = accuracy

    return accuracy, params["max_iter"], loss


study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
study.optimize(objective, n_trials=1000)

# best_params = study.best_params

# print(best_params)
#
# clf = svm.SVC(**best_params)
# clf.fit(X_train, y_train)

# accuracy = clf.score(X_test, y_test)

# print(accuracy)
