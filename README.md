README_final.md
# Water Potability Classification

ML-проект по бинарной классификации пригодности воды для питья на основе физико-химических характеристик.

---

## Задача

Построить модель, которая предсказывает:

- `0` — вода непригодна  
- `1` — вода пригодна  

Тип задачи: **binary classification**

---

## 📊 Данные

- 3276 объектов  
- 9 числовых признаков  
- есть пропуски (`ph`, `Sulfate`, `Trihalomethanes`)  
- умеренный дисбаланс классов (~60/40)

Особенности данных:

- слабая корреляция с target  
- сильное пересечение распределений  
- нет одного доминирующего признака  

---

## Структура проекта

```
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_models.ipynb
├── src/
│   ├── data.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── best_model.pkl
├── requirements.txt
└── README.md
```

---

## 📓 Ноутбуки

### `01_eda.ipynb`

Разведочный анализ данных:

- проверка структуры данных
- анализ пропусков
- распределения признаков
- сравнение классов
- корреляции

Вывод: признаки слабо разделяют классы → нужна нелинейная модель + feature engineering

---

### `02_baseline.ipynb`

Построение baseline:

- DummyClassifier
- LogisticRegression
- RandomForest
- Pipeline + imputation
- cross-validation

Вывод:  
RandomForest — лучший baseline, но низкий recall класса `1`

---

### `03_models.ipynb`

Улучшение модели:

- feature engineering (interaction, ratios, полиномы)
- SMOTE (балансировка классов)
- сравнение моделей:
  - RandomForest
  - CatBoost
  - LightGBM
- hyperparameter tuning (`RandomizedSearchCV`)
- error analysis
- SHAP интерпретация

Финальная модель:  
**RandomForest + Feature Engineering + SMOTE**

---

## Подход

Использовано:

- Pipeline (sklearn + imblearn)
- Median imputation
- Feature Engineering:
  - логарифмы
  - квадраты
  - взаимодействия
  - отношения
- SMOTE
- Cross-validation
- Hyperparameter tuning
- SHAP

---

## Результаты

### Baseline (RandomForest)

- F1: 0.44  
- Recall: 0.34  
- ROC-AUC: 0.66  

### Final Model

- F1: **0.57**  
- Recall: **0.56**  
- ROC-AUC: **0.70**

Основной рост:

- Recall: +0.22  
- F1: +0.13  

---

## Интерпретация

SHAP показал:

- важные признаки: `Sulfate`, `ph`, interaction features  
- engineered признаки дают сильный вклад  
- модель использует комбинацию слабых сигналов  

---

## Выводы

- линейные модели плохо работают  
- основной прирост — за счёт данных, а не модели  
- SMOTE сильно улучшает recall  
- качество ограничено сложностью данных  

---

## src/

Код вынесен в `src/`:

- `data.py` — загрузка данных  
- `train.py` — финальный pipeline  
- `evaluate.py` — метрики  

👉 это делает проект ближе к production-подходу

---

## Запуск

```bash
pip install -r requirements.txt
```

Порядок:

```
01_eda → 02_baseline → 03_models
```

---

## Итог

Проект реализует полный ML pipeline:

```
EDA → Baseline → Feature Engineering → SMOTE → Tuning → SHAP
```

Главный инсайт:

В tabular задачах качество чаще растёт за счёт работы с данными, а не смены модели.