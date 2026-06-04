# Кластеризация изображений транспортных средств (IntelliVision case)

[![GitHub](https://img.shields.io/badge/GitHub-theKerimKerimov-181717?logo=github)](https://github.com/theKerimKerimov/vehicle-image-clustering-intellivision)
[![Kaggle](https://img.shields.io/badge/Kaggle-kerimkerimov-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/kerimkerimov)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/theKerimKerimov/vehicle-image-clustering-intellivision)
[![Unsupervised ML](https://img.shields.io/badge/Task-Unsupervised%20Clustering-9cf)](https://github.com/theKerimKerimov/vehicle-image-clustering-intellivision)
[![Updated](https://img.shields.io/badge/Updated-2026-blue)](https://github.com/theKerimKerimov/vehicle-image-clustering-intellivision)

## 📊 Обзор проекта

**Бизнес‑задача:** исследовать возможность автоматической разметки изображений транспортных средств (тип кузова, цвет, ракурс и т.п.), а также поиска выбросов в большом потоке данных Smart City/Transportation (кейс IntelliVision).

**Техническая задача:** на основе уже вычисленных дескрипторов четырёх сверточных нейросетей:

- выполнить кластеризацию изображений разными алгоритмами;
- сравнить дескрипторы и варианты предобработки по внутренним метрикам и визуализации;
- найти выбросы с помощью алгоритмов поиска аномалий;
- выбрать конфигурацию (дескриптор + предобработка + алгоритм), подходящую для дальнейшей разметки.

**Данные:** 416 314 изображений и четыре набора дескрипторов (по одной строке на изображение) — `efficientnet-b7`, `osnet`, `vdc_color`, `vdc_type`. Изображения относятся к датасету [VeRi-Wild](https://github.com/JDAI-CV/VeRi) (veriwild); дескрипторы подготовлены в рамках учебного кейса IntelliVision.

## 🏗️ Структура проекта

```text
vehicle-image-clustering-intellivision/
├── data/
│   ├── descriptors/                 # Дескрипторы четырёх моделей (*.pickle), не в Git
│   ├── raw_data/                    # Изображения veriwild после распаковки, не в Git
│   ├── images_paths.csv             # В Git: 416 314 путей (1:1 с дескрипторами)
│   └── clustering_results_best.csv  # В Git: разметка подвыборки 80k (см. ниже)
├── docs/
│   └── images/                      # Скриншоты для README (опционально)
├── notebooks/
│   └── vehicle_image_clustering.ipynb
├── LICENSE
├── .gitignore
├── README.md
└── requirements.txt
```

## 📂 Данные

### Что уже в репозитории

| Файл | Содержимое |
|------|------------|
| `data/images_paths.csv` | 416 314 относительных путей к изображениям (`veriwild/...`), согласованы с pickle-дескрипторами |
| `data/clustering_results_best.csv` | Итоговая кластерная разметка для подвыборки 80 000 объектов (см. таблицу ниже) |

После клонирования эти файлы уже лежат в `data/` — отдельно скачивать их не нужно.

### Что нужно добавить локально

Дескрипторы и сами изображения **тяжёлые** и в Git **не входят**:

1. **Дескрипторы** (`*.pickle` в `data/descriptors/`) — из материалов учебного кейса IntelliVision. Если нет доступа к архиву кейса, напишите автору репозитория (контакты ниже).
2. **Изображения VeRi-Wild** — [JDAI-CV/VeRi](https://github.com/JDAI-CV/VeRi). Распакуйте в `data/raw_data/`, чтобы пути совпадали с `images_paths.csv` (`veriwild/...`).

### Размещение в проекте

```text
data/
├── descriptors/          # не в Git — положить вручную
│   ├── efficientnet-b7.pickle
│   ├── osnet.pickle
│   ├── vdc_color.pickle
│   └── vdc_type.pickle
├── raw_data/             # не в Git — распаковать veriwild
│   └── veriwild/
│       └── ...
├── images_paths.csv      # в Git
└── clustering_results_best.csv   # в Git (перезаписывается ноутбуком при Run All)
```

### `clustering_results_best.csv`

| Параметр | Значение |
|----------|----------|
| Дескриптор | `vdc_type` |
| Предобработка | `StandardScaler` + PCA (90% дисперсии) |
| Алгоритм | `AgglomerativeClustering` |
| Число кластеров | 5 (метки 0–4) |
| Столбцы | `path`, `cluster` |

После полного прогона ноутбука с вашими данными файл перезаписывается с теми же правилами.

## 📈 Описание дескрипторов

- **`efficientnet-b7.pickle`** — EfficientNet‑B7 (ImageNet), 2560 признаков.
- **`osnet.pickle`** — OSNet (re-identification), 512 признаков.
- **`vdc_color.pickle`** — регрессия цвета ТС (RGB), обучение частично на VeRi-Wild, 128 признаков.
- **`vdc_type.pickle`** — классификация типа ТС (10 классов), частично на VeRi-Wild, 512 признаков.

Строки во всех pickle и в `images_paths.csv` согласованы (индекс `i` — одно и то же изображение).

## 🛠️ Техническая реализация

### 1. Предобработка и PCA

- **StandardScaler → PCA** и **MinMaxScaler → PCA** (90% дисперсии, макс. 200 компонент).
- Подвыборка **80 000** объектов (`random_state=42`).
- Компоненты после PCA: efficientnet-b7 — 200 (~63% дисперсии), osnet — 153, vdc_color — 69, vdc_type — 20.

### 2. Кластеризация

Алгоритмы: **MiniBatchKMeans**, **GaussianMixture**, **AgglomerativeClustering**.  
Сетка `k ∈ {5, 10, 15, 20, 25, 30}`; метрики: Calinski–Harabasz (выше лучше), Davies–Bouldin (ниже лучше).  
Оптимальное **k = 5** по сетке для всех дескрипторов.  
`minmax` повышает CH; для `vdc_type` финальная разметка в CSV взята с **`std`** из‑за лучшей интерпретируемости кластеров.

### 3. Визуализация

t-SNE на 5000 точек, scatter по кластерам, сетки примеров изображений (9 на кластер).

### 4. Выбросы (DBSCAN)

Подвыборка **25 000** на дескриптор; доли выбросов: vdc_color 0.03%, efficientnet-b7 0.7%, vdc_type 1.29%, osnet 23.78%.

## 🔑 Ключевые выводы

- Лучшие дескрипторы для кластеризации: **`vdc_type`** (тип ТС) и **`vdc_color`** (цвет).
- **`osnet`** — среднее качество; **`efficientnet-b7`** — наименее интерпретируемые кластеры.
- Для поиска кандидатов на выбросы удобен **`osnet` + DBSCAN** (большая доля метки −1 на подвыборке).

## 🚀 Как запустить

### Установка

```bash
git clone https://github.com/theKerimKerimov/vehicle-image-clustering-intellivision.git
cd vehicle-image-clustering-intellivision
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Подготовка данных

См. раздел [Данные](#-данные): `images_paths.csv` и `clustering_results_best.csv` уже в репозитории; дополнительно нужны только pickle в `data/descriptors/` и изображения в `data/raw_data/veriwild/`.

### Запуск ноутбука

```bash
jupyter notebook notebooks/vehicle_image_clustering.ipynb
```

Выполните **Run All** из каталога `notebooks/` (пути `BASE_DIR = Path('..') / 'data'` рассчитаны на эту структуру).

**Память:** дескрипторы подгружаются сразу в подвыборку (`SUBSET_SIZE = 80_000` в первой ячейке раздела 1), без хранения полных 416k DataFrame в RAM. Рекомендуется **≥16 GB RAM**; при `MemoryError` уменьшите `SUBSET_SIZE` (например, до `30_000`) и перезапустите ядро (**Kernel → Restart**).

## 🛠️ Стек

Python 3.10+, pandas, numpy, scikit-learn, scipy, matplotlib, Pillow, Jupyter.

Зависимости с версиями — в [`requirements.txt`](requirements.txt).

## 📋 Pipeline в ноутбуке

1. Загрузка путей и дескрипторов  
2. Подвыборка 80k, масштабирование, IncrementalPCA  
3. Кластеризация (std и minmax), сравнение метрик  
4. t-SNE и визуализация кластеров по изображениям  
5. DBSCAN, анализ выбросов  
6. Экспорт `clustering_results_best.csv` и итоговые выводы  

## 👤 Автор

**Karim** · 2026

[![GitHub](https://img.shields.io/badge/GitHub-theKerimKerimov-181717?logo=github)](https://github.com/theKerimKerimov)<br>
[![Kaggle](https://img.shields.io/badge/Kaggle-kerimkerimov-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/kerimkerimov)<br>
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kerim--kerimov-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kerim-kerimov-79323b400)<br>
[![LeetCode](https://img.shields.io/badge/LeetCode-KerimK-FFA116?logo=leetcode&logoColor=black)](https://leetcode.com/u/KerimK)<br>
[![Telegram](https://img.shields.io/badge/Telegram-@theDagestani-26A5E4?logo=telegram&logoColor=white)](https://t.me/theDagestani)

📍 Москва · ✉️ [k.kerimow@yandex.ru](mailto:k.kerimow@yandex.ru)

## 📄 Лицензия

Код проекта — [MIT](LICENSE). Данные VeRi-Wild и дескрипторы кейса IntelliVision распространяются по правилам их правообладателей; в репозиторий не включены.
