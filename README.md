# Прогноз кредитного дефолта

Модель оценивает вероятность дефолта в следующем месяце по данным клиента кредитной карты.
Проект включает подготовку данных, подбор моделей, MLflow, DVC, FastAPI и мониторинг PSI.

## Данные и подготовка

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients), авторы I-Cheng Yeh и Che-hui Lien, лицензия CC BY 4.0.
Исходный CSV: 30 000 клиентов, 23 признака, целевая переменная `default.payment.next.month`.
Класс 1 означает дефолт. Идентификатор клиента исключён из признаков.

Обучающая выборка содержит 23 999 строк, тестовая - 6 001. Одинаковые нормализованные
профили клиентов объединены в группы и не пересекают train/test и folds кросс-валидации.
Great Expectations проверяет состав столбцов, типы, пропуски, диапазоны и допустимые коды.
Правила находятся в `data/expectations/`.

Feature Engineering добавляет агрегаты задержек, счетов и платежей, отношения к кредитному
лимиту и возрастную группу. Sklearn Pipeline включает Imputer, StandardScaler,
OneHotEncoder и классификатор. Предобработка обучается внутри каждого CV fold.

## Структура

- `src/pd_model/` - подготовка данных, признаки, обучение, API и мониторинг.
- `tests/` - тесты обработки данных, валидации, моделей, API и PSI.
- `notebooks/01_eda.ipynb` - EDA с графиками и выводами.
- `data/` и `models/` - данные и модели под управлением DVC.
- `config/experiments.yaml` и `params.yaml` - настройки экспериментов и пайплайна.
- `reports/` - метрики, графики, сравнение моделей и презентация.
- `scripts/` - восстановление архивов, проверка API и запуск Docker.
- `.github/workflows/` - CI и публикация контейнера.

## Установка

Нужны Python 3.12, Git и Make. Команды выполняются из папки проекта.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

CSV и модели не хранятся в Git. Скачайте `dvc-remote.tar.gz` из раздела
[Releases](https://github.com/fire57/PdModel/releases) в папку `artifacts/` и восстановите их:

```bash
python scripts/bootstrap_dvc_remote.py --archive artifacts/dvc-remote.tar.gz
dvc status
```

## Пайплайн и тесты

```bash
make quality
make test
make validate
make repro
```

DVC-пайплайн состоит из стадий `prepare`, `train`, `evaluate`. Неизменённые стадии пропускаются.
Для полного повторного обучения:

```bash
dvc repro --force
```

Негативная проверка данных:

```bash
python -m pd_model.data.validation --data examples/anomalous_credit_sample.csv --suite raw
```

В примере некорректный возраст. Валидатор должен завершиться с кодом 2.

## Модели и результаты

E0 - DummyClassifier. E1 - LogisticRegression на исходных признаках.
E2 - LogisticRegression с агрегатами и GridSearchCV. E3 - RandomForest с агрегатами,
E4 - GradientBoosting на исходных признаках, E5 - GradientBoosting с агрегатами.
Для E3-E5 используется RandomizedSearchCV. Кросс-валидация: пять групповых folds, seed 42.
Модель выбирается по среднему CV ROC-AUC. Test не участвует в подборе параметров и выборе модели.

| Модель | CV ROC-AUC | Test ROC-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| E1 | 0.7691 | 0.7716 | 0.7001 | 0.3622 | 0.4774 |
| E2 | 0.7764 | 0.7781 | 0.4733 | 0.6130 | 0.5341 |
| E3 | 0.7847 | 0.7905 | 0.4865 | 0.6122 | 0.5422 |
| E4 | 0.7805 | 0.7870 | 0.6945 | 0.3697 | 0.4826 |
| E5 | 0.7848 | 0.7942 | 0.6961 | 0.3795 | 0.4912 |

Выбрана E5. Её преимущество над E3 по CV ROC-AUC около 0.00019 и не доказывает
статистически значимого превосходства. E3 даёт более высокие F1 и Recall при том же пороге 0.5.
Добавление агрегатов в E5 сопровождается ростом CV ROC-AUC относительно E4.
В E1/E2 меняются и признаки, и подбор параметров, поэтому их эффекты не разделены.

Подробные результаты: [сравнение моделей](reports/experiments/test_comparison.csv),
[метрики](reports/metrics/test_metrics.json), [ROC-кривая E5](reports/figures/evaluation/E5_gb_features_roc.png).

## MLflow

Для восстановления истории экспериментов скачайте `mlflow-tracking.tar.gz` из Releases
в `artifacts/`. Восстановление выполняется при отсутствии локальных `mlflow.db` и `mlartifacts/`:

```bash
python scripts/archive_tracking.py restore
make mlflow
```

Откройте http://127.0.0.1:5000, выберите `Credit_Default_Prediction`, запуски E1-E5 и Compare.
В MLflow сохраняются параметры, CV/test-метрики, результаты поиска, графики и обученные Pipeline
через `mlflow.sklearn.log_model`.

## API

```bash
make api
```

Документация API: http://127.0.0.1:8000/docs. Endpoint `/predict` принимает JSON
с 23 исходными признаками и возвращает `default_prediction` и `default_probability`.
Пример запроса:

```bash
curl --fail -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @examples/sample_request.json
```

Проверка API выполняется во втором терминале с активированной средой:

```bash
make smoke
```

## Docker

При запущенном Docker Desktop/Engine:

```bash
docker build --tag pd-model:local .
docker run --detach --name pd-model --publish 127.0.0.1:8001:8000 pd-model:local
python scripts/smoke_api.py --url http://127.0.0.1:8001
```

Имя `pd-model` и порт 8001 должны быть свободны.
Скрипт `bash scripts/docker_local.sh` выполняет сборку и запуск на порту 8000.

## Мониторинг

```bash
make monitor API_URL=http://127.0.0.1:8001
make monitor-shift API_URL=http://127.0.0.1:8001
```

Симулятор отправляет по 1 000 строк тестовой выборки в API и сравнивает признаки
и вероятности с train reference. Во втором сценарии `LIMIT_BAL` уменьшается в 20 раз.
PSI рассчитывается по фиксированным train bins. Пороги: warning 0.1, alert 0.25.
Отчёты и графики сохраняются в `reports/monitoring/`.

## CI

GitHub Actions запускает Black, flake8, pytest, восстановление DVC, Great Expectations,
пробное обучение и проверку API. Для восстановления задайте repository variable
`PD_MODEL_DVC_ARCHIVE_URL` со ссылкой на архив DVC из Releases.
Workflow контейнера запускается вручную и публикует образ после проверки API и мониторинга.

[Презентация](reports/presentation/Презентация.pptx).
