import argparse
import logging
import yaml
import mlflow
import optuna
import os
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, hinge_loss
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', type=int, required=True)
parser.add_argument('--logs', type=bool, required=True)
args = parser.parse_args()
max_iter = args.max_iter
logs = args.logs
if not logs:
    logging.getLogger('mlflow').setLevel(logging.ERROR)

df = pd.read_csv("datasets/prepared_data.csv")

labels = list(map(int, df["labels"]))
df = df.drop(['labels'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, random_state=42)
kernel = ['linear', 'poly', 'rbf', 'sigmoid']


# mlflow.set_tracking_uri("http://localhost:5000")


def objective(trial, **kwargs):
    params = {
        "C": trial.suggest_loguniform('C', 1e-5, 100),
        "gamma": trial.suggest_loguniform('gamma', 1e-5, 10),
        "kernel": trial.suggest_categorical("kernel", kernel),
        # "degree": trial.suggest_int("degree", low=3, high=5),
        "shrinking": trial.suggest_int("shrinking", low=0, high=1),
        "class_weight": trial.suggest_int("class_weight", low=2, high=8),
        "probability": trial.suggest_int("probability", low=0, high=1)
    }

    mlflow.set_experiment(f"my_experiment_maxiter_{max_iter}_{trial.number}")
    with mlflow.start_run():
        mlflow.sklearn.autolog()
        mlflow.log_param("C", params["C"])
        mlflow.log_param("gamma", params["gamma"])
        mlflow.log_param("kernel", params["kernel"])
        mlflow.log_param("shrinking", False if params["shrinking"] == 0 else True)
        mlflow.log_param("class_weight", params["class_weight"])
        mlflow.log_param("probability", False if params["probability"] == 0 else True)

        clf = svm.SVC(C=params["C"],  # размер штрафа за неправильную классификацию
                      gamma=params["gamma"],
                      # параметр, влияющий на значимость одиночно стоящих точек для определения границ
                      random_state=42,
                      kernel=params["kernel"],  # тип ядра
                      # degree=params["degree"], # степень для полиномиального ядра
                      shrinking=False if params["shrinking"] == 0 else True,  # метод для сокращения вычислений
                      max_iter=max_iter,
                      class_weight={0: 1, 1: params["class_weight"]},
                      probability=False if params["probability"] == 0 else True
                      )

        clf.fit(X_train.values, y_train)

        accuracy = clf.score(X_test.values, y_test)
        predicts = clf.predict(X_test.values)

        results = {"report": classification_report(predicts, y_test),
                   "loss": hinge_loss(predicts, y_test),
                   "accuracy": accuracy}

        mlflow.log_metric("lol", 1)
        mlflow.log_metric("loss", hinge_loss(predicts, y_test))
        mlflow.log_metric("accuracy", accuracy)

    return accuracy
    # return accuracy, params["max_iter"], loss


# study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
study = optuna.create_study(directions=['maximize'])
study.optimize(objective, n_trials=1)
best_params = study.best_params
best_params["max_iter"] = max_iter
print(best_params)

with open(f'spec_params_{max_iter}_max_iter.yaml', 'w') as f:
    yaml.dump({f'best_params_max_iter_{max_iter}': best_params}, f)

# save best_params params

# if os.path.isfile('spec_params.yaml'):
#
#     with open('spec_params.yaml', 'r') as f:
#         config = yaml.safe_load(f)
#     print(config)
#     if config is None:
#         config = {}
#     config[f'best_params_max_iter_{max_iter}'] = best_params
# else:
#     config = {f'best_params_max_iter_{max_iter}': best_params}
#
# with open(f'spec_params_{max_iter}_max_iter.yaml', 'w') as f:
#     yaml.dump(config, f)
