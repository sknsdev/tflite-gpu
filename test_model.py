#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестирование обученной модели TensorFlow Lite
"""

import numpy as np
try:
    # Попытка использовать новый LiteRT API
    from ai_edge_litert.interpreter import Interpreter
    USING_LITERT = True
except ImportError:
    # Fallback на старый TensorFlow Lite API
    import tensorflow as tf
    from tensorflow.lite.python.interpreter import Interpreter
    USING_LITERT = False

from PIL import Image
import json
import os
import sys

class CardPredictor:
    def __init__(self, model_path="/app/card_recognition_model.tflite", 
                 class_names_path="/app/class_names.json"):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.interpreter = None
        self.class_names = []
        self.input_details = None
        self.output_details = None
        
        self.load_model()
        self.load_class_names()
    
    def load_model(self):
        """Загрузка TensorFlow Lite модели"""
        if not os.path.exists(self.model_path):
            print(f"Ошибка: модель не найдена по пути {self.model_path}")
            return False
            
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Модель загружена: {self.model_path}")
        print(f"Входной тензор: {self.input_details[0]['shape']}")
        print(f"Выходной тензор: {self.output_details[0]['shape']}")
        return True
    
    def load_class_names(self):
        """Загрузка названий классов"""
        if not os.path.exists(self.class_names_path):
            print(f"Ошибка: файл с названиями классов не найден: {self.class_names_path}")
            return False
            
        with open(self.class_names_path, 'r', encoding='utf-8') as f:
            self.class_names = json.load(f)
            
        print(f"Загружено {len(self.class_names)} классов")
        return True
    
    def preprocess_image(self, image_path):
        """Предобработка изображения"""
        try:
            # Загрузка и изменение размера изображения
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            
            # Проверяем тип входного тензора
            input_dtype = self.input_details[0]['dtype']
            
            if input_dtype == np.uint8:
                # Для UINT8 модели - без нормализации
                img_array = np.array(img, dtype=np.uint8)
            else:
                # Для FLOAT32 модели - с нормализацией
                img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Добавление batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            print(f"Ошибка обработки изображения {image_path}: {e}")
            return None
    
    def predict(self, image_path):
        """Предсказание класса карты"""
        if self.interpreter is None:
            print("Модель не загружена")
            return None
            
        # Предобработка изображения
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None
        
        # Установка входных данных
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        
        # Выполнение предсказания
        self.interpreter.invoke()
        
        # Получение результата
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output_data[0]
        
        # Получение топ-3 предсказаний
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            confidence = predictions[idx]
            class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            results.append((class_name, confidence))
        
        return results
    
    def test_on_sample_images(self, cards_dir="/app/cards", num_samples=5):
        """Тестирование на случайных изображениях из датасета"""
        print("\nТестирование модели на случайных изображениях...")
        
        # Сбор всех изображений
        all_images = []
        for root, dirs, files in os.walk(cards_dir):
            for file in files:
                if file.endswith('.webp'):
                    all_images.append(os.path.join(root, file))
        
        if len(all_images) == 0:
            print("Не найдено изображений для тестирования")
            return
        
        # Случайный выбор изображений
        np.random.seed(42)
        sample_images = np.random.choice(all_images, min(num_samples, len(all_images)), replace=False)
        
        for img_path in sample_images:
            print(f"\nТестирование: {img_path}")
            
            # Извлечение истинного класса из пути
            path_parts = img_path.replace('\\', '/').split('/')
            set_name = path_parts[-3]
            card_folder = path_parts[-2]
            
            # Определение типа карты
            card_type = "normal"
            if "_alt" in card_folder:
                card_type = "alt"
                card_number = card_folder.replace("_alt", "")
            elif "_pf" in card_folder:
                card_type = "pf"
                card_number = card_folder.replace("_pf", "")
            else:
                card_number = card_folder
            
            true_class = f"{set_name}_{card_number}_{card_type}"
            print(f"Истинный класс: {true_class}")
            
            # Предсказание
            results = self.predict(img_path)
            if results:
                print("Предсказания:")
                for i, (class_name, confidence) in enumerate(results, 1):
                    print(f"  {i}. {class_name}: {confidence:.4f}")
                
                # Проверка правильности
                if results[0][0] == true_class:
                    print("✓ Правильное предсказание!")
                else:
                    print("✗ Неправильное предсказание")
            else:
                print("Ошибка предсказания")

def main():
    """Основная функция"""
    predictor = CardPredictor()
    
    if len(sys.argv) > 1:
        # Тестирование на конкретном изображении
        image_path = sys.argv[1]
        print(f"Тестирование изображения: {image_path}")
        
        results = predictor.predict(image_path)
        if results:
            print("\nРезультаты предсказания:")
            for i, (class_name, confidence) in enumerate(results, 1):
                print(f"{i}. {class_name}: {confidence:.4f}")
        else:
            print("Ошибка предсказания")
    else:
        # Тестирование на случайных изображениях
        predictor.test_on_sample_images()

if __name__ == "__main__":
    main()