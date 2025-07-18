#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Card Recognition Model Training Script
Тренировка модели для распознавания карт коллекционной карточной игры
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from PIL import Image
import json

# Настройка GPU
print("Проверка доступности GPU...")
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU найден: {physical_devices[0]}")
else:
    print("GPU не найден, используется CPU")

class CardDatasetLoader:
    def __init__(self, cards_dir="/app/cards", img_size=(224, 224)):
        self.cards_dir = cards_dir
        self.img_size = img_size
        self.class_names = []
        self.class_to_idx = {}
        
    def load_dataset(self):
        """Загрузка датасета карт"""
        images = []
        labels = []
        
        # Проходим по всем сетам
        for set_name in os.listdir(self.cards_dir):
            set_path = os.path.join(self.cards_dir, set_name)
            if not os.path.isdir(set_path):
                continue
                
            print(f"Обработка сета: {set_name}")
            
            # Проходим по всем картам в сете
            for card_folder in os.listdir(set_path):
                card_path = os.path.join(set_path, card_folder)
                if not os.path.isdir(card_path):
                    continue
                    
                # Определяем тип карты и номер
                card_type = "normal"  # по умолчанию
                if "_alt" in card_folder:
                    card_type = "alt"
                    card_number = card_folder.replace("_alt", "")
                elif "_pf" in card_folder:
                    card_type = "pf"
                    card_number = card_folder.replace("_pf", "")
                else:
                    card_number = card_folder
                    
                # Формируем метку класса
                class_label = f"{set_name}_{card_number}_{card_type}"
                
                if class_label not in self.class_to_idx:
                    self.class_to_idx[class_label] = len(self.class_names)
                    self.class_names.append(class_label)
                    
                # Загружаем все изображения карты
                image_files = glob.glob(os.path.join(card_path, "*.webp"))
                for img_file in image_files:
                    try:
                        img = Image.open(img_file).convert('RGB')
                        img = img.resize(self.img_size)
                        img_array = np.array(img) / 255.0
                        
                        images.append(img_array)
                        labels.append(self.class_to_idx[class_label])
                    except Exception as e:
                        print(f"Ошибка загрузки {img_file}: {e}")
                        
        print(f"Загружено {len(images)} изображений, {len(self.class_names)} классов")
        return np.array(images), np.array(labels)
    
    def save_class_names(self, filepath="/app/class_names.json"):
        """Сохранение названий классов"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.class_names, f, ensure_ascii=False, indent=2)

def create_model(num_classes, input_shape=(224, 224, 3)):
    """Создание модели CNN для классификации карт"""
    model = keras.Sequential([
        # Входной слой
        layers.Input(shape=input_shape),
        
        # Аугментация данных
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Сверточные слои
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu'),
        layers.MaxPooling2D(),
        
        # Глобальное усреднение
        layers.GlobalAveragePooling2D(),
        
        # Полносвязные слои
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    """Основная функция обучения"""
    print("Начало обучения модели...")
    
    # Загрузка данных
    loader = CardDatasetLoader()
    X, y = loader.load_dataset()
    
    if len(X) == 0:
        print("Ошибка: не найдено изображений для обучения")
        return
    
    # Сохранение названий классов
    loader.save_class_names()
    
    # Разделение на train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер валидационной выборки: {len(X_val)}")
    print(f"Количество классов: {len(loader.class_names)}")
    
    # Создание модели
    model = create_model(len(loader.class_names))
    
    # Компиляция модели
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Вывод архитектуры модели
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Обучение модели
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Оценка модели
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Точность на валидации: {val_accuracy:.4f}")
    
    # Конвертация в TensorFlow Lite
    print("Конвертация в TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Квантизация для мобильных устройств
    def representative_dataset():
        for i in range(100):
            yield [X_train[i:i+1].astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    # Сохранение модели
    with open('/app/card_recognition_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Модель сохранена как card_recognition_model.tflite")
    print(f"Размер модели: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    return model, history

if __name__ == "__main__":
    train_model()