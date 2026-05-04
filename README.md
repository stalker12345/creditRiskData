<div align="center">
  <h1>CreditRiskData</h1>
  <p><strong>ML-система оценки кредитного риска с интерактивным Streamlit-интерфейсом</strong></p>
  <p>
    Проект объединяет исследовательский notebook, обученную модель и готовое веб-приложение
    для оценки вероятности дефолта по кредитной заявке.
  </p>
  <p>
    <a href="https://creditriskdata.streamlit.app/">
      <img src="https://img.shields.io/badge/Live%20Demo-Open%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
    </a>
    <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/scikit--learn-Gradient%20Boosting-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  </p>
  <p>
    <img src="https://img.shields.io/badge/ROC--AUC-0.949-0A7E8C?style=flat-square" alt="ROC-AUC">
    <img src="https://img.shields.io/badge/F1-0.842-0A7E8C?style=flat-square" alt="F1">
    <img src="https://img.shields.io/badge/Recall-75.4%25-0A7E8C?style=flat-square" alt="Recall">
    <img src="https://img.shields.io/badge/Precision-95.2%25-0A7E8C?style=flat-square" alt="Precision">
  </p>
</div>

## Live Demo

**Приложение уже доступно онлайн:**  
https://creditriskdata.streamlit.app/

## О проекте

`CreditRiskData` решает практическую задачу кредитного скоринга: по параметрам заёмщика и кредита система оценивает вероятность дефолта и даёт понятную бизнес-интерпретацию результата.

Проект включает:

- исследовательский notebook с полной цепочкой подготовки данных, обучения и сохранения артефактов;
- обученную модель `GradientBoostingClassifier`;
- готовое Streamlit-приложение для ввода заявки и получения решения;
- метаданные, метрики и параметры преобразований для корректного инференса.

## Что умеет приложение

- оценивать риск дефолта по одной заявке в интерактивной форме;
- показывать вероятности возврата и дефолта;
- визуализировать результат в интерфейсе;
- объяснять решение в бизнес-терминах;
- выводить ключевые метрики качества модели.

## Ключевые метрики модели

| Метрика | Значение |
|---|---:|
| Accuracy | 93.81% |
| Precision | 95.25% |
| Recall | 75.45% |
| F1-score | 0.8420 |
| ROC-AUC | 0.9487 |
| PR-AUC | 0.9093 |
| MCC | 0.8126 |
| Specificity | 98.95% |

**Матрица ошибок**

| Факт \\ Прогноз | Non-default | Default |
|---|---:|---:|
| Non-default | 3759 | 40 |
| Default | 261 | 802 |

## Архитектура решения

1. Пользователь вводит исходные данные заёмщика в Streamlit-интерфейсе.
2. `load_model.py` валидирует вход, применяет нужные преобразования и загружает pipeline.
3. Модель считает вероятности классов `default / non-default`.
4. Приложение визуализирует результат, показывает рекомендацию и важные факторы риска.

### Преобразования признаков

- `person_income` подаётся в исходном формате и логарифмируется перед предсказанием;
- для ряда числовых признаков применяется Box-Cox с сохранёнными параметрами;
- pipeline выполняет масштабирование и кодирование категориальных признаков.

## Быстрый старт

### Локальный запуск

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r streamlit_model/requirements_streamlit.txt
streamlit run app.py
```

После запуска приложение будет доступно по адресу `http://localhost:8501`.

### Запуск через готовый скрипт

```bash
./run_streamlit.sh
```

## Структура проекта

```text
CreditRiskData/
├── app.py                           # Streamlit-приложение
├── streamlit_model/
│   ├── model.skops                  # Сохранённая модель
│   ├── load_model.py                # Загрузка модели и инференс
│   ├── metadata.json                # Метаданные признаков
│   ├── model_metrics.json           # Метрики качества
│   ├── example_data.json            # Примеры входных данных
│   ├── lambdas.pkl                  # Параметры Box-Cox
│   └── requirements_streamlit.txt   # Зависимости для Streamlit
├── сreditRiskDataset.ipynb          # Исследование, обучение и экспорт артефактов
├── QUICKSTART.md                    # Короткая инструкция по запуску
├── README_STREAMLIT.md              # Расширенное описание интерфейса
└── run_streamlit.sh                 # Скрипт запуска
```

## Технологический стек

- Python
- Streamlit
- pandas
- NumPy
- SciPy
- scikit-learn
- skops
- Plotly
- joblib

## Для чего этот проект полезен

- как демонстрация полного ML-пайплайна от анализа до деплоя;
- как учебный пример кредитного скоринга;
- как база для доработки в сторону production scoring-сервиса;
- как витрина дипломного проекта с live demo.

## Дополнительно

- Полный интерактивный интерфейс: `app.py`
- Быстрый запуск: `QUICKSTART.md`
- Развёрнутая документация по приложению: `README_STREAMLIT.md`
- Онлайн-версия: https://creditriskdata.streamlit.app/

## Лицензия

Проект распространяется в образовательных целях. Подробности указаны в файле `LICENSE`.
