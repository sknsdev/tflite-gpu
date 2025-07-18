# Card Recognition Model

Проект для распознавания карт коллекционной карточной игры с использованием TensorFlow Lite.

## Описание

Модель обучается распознавать карты и возвращать их название в формате:
`номерсета_номеркарты_версия карты`

Пример: `laar_1_normal`, `s3_126_alt`

### Поддерживаемые сеты:
- s1, s2, s3, s4, s5, laar

### Типы карт:
- `normal` - обычные карты
- `alt` - альтернативные карты  
- `pf` - полноформатные карты

## Структура проекта

```
├── cards/                    # Датасет с изображениями карт
│   └── laar/                # Папка сета
│       ├── 1/               # Папка карты (номер)
│       ├── 2_alt/           # Папка карты с типом
│       └── ...
├── train_model.py           # Скрипт обучения модели
├── test_model.py            # Скрипт тестирования модели
├── requirements.txt         # Зависимости Python
├── Dockerfile              # Docker конфигурация
└── README.md               # Этот файл
```

## Запуск проекта

### Требования
- Docker с поддержкой GPU
- NVIDIA GPU (протестировано на RTX 3070)
- WSL2 (для Windows)
- NVIDIA Container Toolkit

### Быстрый старт

**Способ 1: Использование скрипта управления (рекомендуется)**

```bash
# Проверка статуса проекта
python run.py status

# Полный цикл: сборка -> проверка -> обучение -> тестирование
python run.py all

# Или по шагам:
python run.py build    # Сборка образа
python run.py check    # Проверка GPU
python run.py train    # Обучение модели
python run.py test     # Тестирование
```

**Способ 2: Docker команды напрямую**

1. **Сборка Docker образа:**
```bash
docker build -t card-recognition .
```

2. **Проверка GPU:**
```bash
docker run --rm --gpus all -v %cd%:/app card-recognition python check_gpu.py
```

3. **Запуск обучения модели:**
```bash
docker run -it --gpus all -v %cd%:/app --name card-training card-recognition python train_model.py
```

4. **Тестирование модели:**
```bash
docker run --rm --gpus all -v %cd%:/app card-recognition python test_model.py
```

5. **Тестирование на конкретном изображении:**
```bash
docker run --rm --gpus all -v %cd%:/app card-recognition python test_model.py /app/cards/laar/1/laar_1.webp
```

**Способ 3: Docker Compose**

```bash
# Проверка окружения
docker-compose run card-recognition

# Обучение модели
docker-compose run train

# Тестирование
docker-compose run test
```

### Команда из задания

Как указано в задании, вы можете использовать:
```bash
docker run -it --gpus all -v absolute_path/src:/app/ --name container_name tensorflow/tensorflow:latest-gpu
```

Затем внутри контейнера:
```bash
pip install -r requirements.txt
python train_model.py
```

## Результаты

После обучения будут созданы файлы:
- `card_recognition_model.tflite` - обученная модель для мобильных устройств
- `class_names.json` - список классов карт

## Особенности модели

- Использует сверточную нейронную сеть (CNN)
- Оптимизирована для мобильных устройств (TensorFlow Lite)
- Поддерживает квантизацию INT8 для уменьшения размера
- Включает аугментацию данных для улучшения обобщения
- Использует GPU для ускорения обучения

## Архитектура модели

- Входной размер: 224x224x3 (RGB изображения)
- 4 сверточных блока с MaxPooling
- GlobalAveragePooling для уменьшения параметров
- 2 полносвязных слоя с Dropout
- Softmax активация для классификации

## Производительность

Модель оптимизирована для:
- Быстрого инференса на мобильных устройствах
- Малого размера файла модели
- Высокой точности распознавания карт