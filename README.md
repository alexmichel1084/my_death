### Варианты задачи

решающая модель линейный SVM (scikit-learn), задача классификации Y, датасет: Кредитный скорринг (bank.csv)


### Постановка задачи

Данная инструкция предоставляет подробное руководство по обучению модели на основе предоставленного датасета. Процесс разделен на три этапа: обработка данных, выбор признаков и обучение модели.

Весь необходимый код должен быть размещен в репозитории, а установка всех зависимостей и запуск кода должны выполняться командами:

```bash
pip install -r requirements.txt
dvc repro
mlflow ui
```



#### Обработка данных (использование lazy API в polars)

- Преобразование всех категориальных признаков с использованием Target Encoding (для Linear SVM). 
- Замена значений "none" средними значениями (без удаления соответствующих столбцов).
- Преобразование всех числовых признаков с использованием Z-score.
- Фильтрация данных на наличие корреляции, удаляя признаки, коррелированные более чем на 0.5 (для Linear SVM).

Подробно можно посмотреть в cli/preprocessing.py


**На входе:** датасет

**На выходе:** предобработанный датасет


#### Выбор признаков (использование Optuna)

Отбор наиболее полезных признаков. Эксперименты запускаются с различными наборами входных данных, и метрики вычисляются на валидационных данных.

**На входе:** предобработанный датасет

**На выходе:** CSV-файл, отображающий выбранные признаки и соответствующие метрики для каждого проведенного эксперимента

Примечание: автор задания просит на выходе из данного этапа csv, а на следующем этапе параметры запуска определяются в spec_params.yaml.
Я решил, что разумнее на этом этапе сохранять best_trials сразу в spec_params.

Точнее я делаю два разных запуска подбора гиперпараметров для разных max_iter и добавляю промежуточный stage, который мёрджит результаты в spec.yaml

Подробнее код можно посмотреть в cli/optuna_best_params

P.s.: На всякий случай я логирую все запуски, а не только лучшие из следующего пункта. Привычка с работы.

#### Обучение модели (использование DVC и MLflow)

На последнем этапе необходимо логировать все гиперпараметры модели и метрики на тестовых данных с использованием MLflow. Параметры стадии определяются в файле `spec_params.yaml`, а загрузка происходит через переменные в `dvc.yaml`. Загруженные параметры передаются в стадию обучения с использованием `matrix` (для Random Forest) и `foreach do` (для Linear SVM).

Параметризация стадии:
- Все параметры стадии подробно описаны в файле `spec_params.yaml`.
    - Для Linear SVM предусмотрен список с различными значениями `max_iter` и `loss`.
- Загрузка всех параметров осуществляется через `vars` в файле `dvc.yaml` с использованием `spec_params.yaml`.
- Загруженные параметры из файла `spec_params.yaml` передаются в стадию обучения (`train`) с использованием:
    - `foreach do` для Linear SVM.

**На входе:** предобработанный датасет, yaml-файл

**На выходе:** модель, CSV-файл с метриками для каждой модели отдельно


