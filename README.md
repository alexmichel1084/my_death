### Варианты задачи

1. решающая модель Random Forest (scikit-learn), задача регрессии MathScore, датасет: Успеваемость студентов (edu.csv)
2. решающая модель линейный SVM (scikit-learn), задача классификации Y, датасет: Кредитный скорринг (bank.csv)


### Постановка задачи

Данная инструкция предоставляет подробное руководство по обучению модели на основе предоставленного датасета. Процесс разделен на три этапа: обработка данных, выбор признаков и обучение модели.

Весь необходимый код должен быть размещен в репозитории, а установка всех зависимостей и запуск кода должны выполняться командами:

```bash
pip install -r requirements.txt
dvc repro
mlflow ui
```

Заметки:
- Для DVC каждая стадия должна быть строго описана в формате YAML с обязательными полями `deps` и `outs`, которые должны учесть все результаты и зависимости на каждом этапе.
- Перед запуском проекта в новом виртуальном окружении убедитесь, что команды `pip install -r requirements.txt`, `dvc repro`, и `mlflow ui` успешно выполняются. Это важно для того, чтобы преподаватель мог успешно запустить все этапы и провести проверку.
- Для повышение оценки рекомендуется провести EDA с использованием Polars.

#### Обработка данных (использование lazy API в polars)

На этом этапе обязательно использовать только lazy API из библиотеки Polars для обработки данных. Процедуры обработки данных включают следующие шаги:

- Преобразование всех категориальных признаков с использованием Count Encoding (для Random Forest).
- Преобразование всех категориальных признаков с использованием Target Encoding (для Linear SVM).
- Замена значений "none" средними значениями (без удаления соответствующих столбцов).
- Преобразование всех числовых признаков с использованием Z-score.
- Фильтрация данных на наличие корреляции, удаляя признаки, коррелированные более чем на 0.5 (для Linear SVM).
- Удаление шумных значений, находящихся за пределами 5 и 95 перцентилей (для Random Forest).

**На входе:** датасет

**На выходе:** предобработанный датасет

Заметка: Вся обработка данных должна быть выполнена с использованием lazy API в библиотеке Polars, без использования Pandas. Так же на этой стадии нельзя использовать SKLearn (StandardScaler), все через Polars.

#### Выбор признаков (использование Optuna)

На этом этапе необходимо использовать библиотеку Optuna для отбора наиболее полезных признаков. Эксперименты запускаются с различными наборами входных данных, и метрики вычисляются на валидационных данных.

**На входе:** предобработанный датасет

**На выходе:** CSV-файл, отображающий выбранные признаки и соответствующие метрики для каждого проведенного эксперимента

#### Обучение модели (использование DVC и MLflow)

На последнем этапе необходимо логировать все гиперпараметры модели и метрики на тестовых данных с использованием MLflow. Параметры стадии определяются в файле `spec_params.yaml`, а загрузка происходит через переменные в `dvc.yaml`. Загруженные параметры передаются в стадию обучения с использованием `matrix` (для Random Forest) и `foreach do` (для Linear SVM).

Параметризация стадии:
- Все параметры стадии подробно описаны в файле `spec_params.yaml`.
    - Для Random Forest предусмотрены списки с различными значениями `n_estimators` и `max_depth`.
    - Для Linear SVM предусмотрен список с различными значениями `max_iter` и `loss`.
- Загрузка всех параметров осуществляется через `vars` в файле `dvc.yaml` с использованием `spec_params.yaml`.
- Загруженные параметры из файла `spec_params.yaml` передаются в стадию обучения (`train`) с использованием:
    - `matrix` для Random Forest.
    - `foreach do` для Linear SVM.

**На входе:** предобработанный датасет, CSV-файл с выбранными признаками

**На выходе:** модель, CSV-файл с метриками для каждой модели отдельно



### Сроки задачи и оценка
На задачу даётся 12 дней (по 4 дня на каждую стадию), ее необходимо сдать до 23:59 27.10.2023. После этого задача будет проверена, если задача выполнена успешно, будет назначена защита. Успешная задача - это задача, в которой все этапы выполнены как описано. При защите необходимо будет продемонстрировать работу пайплайна, ответить на вопросы по проекту и по курсу (как было ранее).


### Если что-то не так
Не понятна задача, не понятно описание, проблеммы с данными,проблеммы с моделями,проблеммы с фреймворками - пишите мне в телеграм для консультации, чтобы не терять время. @kudep