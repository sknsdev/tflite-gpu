#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрация работы обученной модели распознавания карт
"""

import os
import json
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

from PIL import Image, ImageDraw, ImageFont
import random
import sys

class CardRecognitionDemo:
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
            print(f"❌ Модель не найдена: {self.model_path}")
            print("Сначала обучите модель: python train_model.py")
            return False
            
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✅ Модель загружена: {self.model_path}")
        model_size = os.path.getsize(self.model_path) / 1024 / 1024
        print(f"📊 Размер модели: {model_size:.2f} MB")
        return True
    
    def load_class_names(self):
        """Загрузка названий классов"""
        if not os.path.exists(self.class_names_path):
            print(f"❌ Файл классов не найден: {self.class_names_path}")
            return False
            
        with open(self.class_names_path, 'r', encoding='utf-8') as f:
            self.class_names = json.load(f)
            
        print(f"📋 Загружено {len(self.class_names)} классов карт")
        return True
    
    def preprocess_image(self, image_path):
        """Предобработка изображения для модели"""
        try:
            img = Image.open(image_path).convert('RGB')
            original_size = img.size
            
            # Изменение размера для модели
            img_resized = img.resize((224, 224))
            
            # Проверяем тип входного тензора
            input_dtype = self.input_details[0]['dtype']
            
            if input_dtype == np.uint8:
                # Для UINT8 модели - без нормализации
                img_array = np.array(img_resized, dtype=np.uint8)
            else:
                # Для FLOAT32 модели - с нормализацией
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, img, original_size
        except Exception as e:
            print(f"❌ Ошибка обработки изображения {image_path}: {e}")
            return None, None, None
    
    def predict_card(self, image_path, top_k=3):
        """Предсказание класса карты"""
        if self.interpreter is None:
            print("❌ Модель не загружена")
            return None
            
        # Предобработка
        img_array, original_img, original_size = self.preprocess_image(image_path)
        if img_array is None:
            return None
        
        # Инференс
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        
        # Получение результатов
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output_data[0]
        
        # Топ-K предсказаний
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            confidence = predictions[idx]
            class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            results.append({
                'class': class_name,
                'confidence': float(confidence),
                'percentage': float(confidence * 100)
            })
        
        return {
            'predictions': results,
            'image_path': image_path,
            'original_image': original_img,
            'original_size': original_size
        }
    
    def parse_card_name(self, class_name):
        """Разбор названия карты на компоненты"""
        parts = class_name.split('_')
        if len(parts) >= 3:
            return {
                'set': parts[0],
                'number': parts[1], 
                'type': parts[2]
            }
        return {'set': 'unknown', 'number': 'unknown', 'type': 'unknown'}
    
    def format_prediction_text(self, prediction):
        """Форматирование текста предсказания"""
        card_info = self.parse_card_name(prediction['class'])
        
        type_names = {
            'normal': 'Обычная',
            'alt': 'Альтернативная', 
            'pf': 'Полноформатная'
        }
        
        type_display = type_names.get(card_info['type'], card_info['type'])
        
        return f"""Сет: {card_info['set'].upper()}
Карта: #{card_info['number']}
Тип: {type_display}
Уверенность: {prediction['percentage']:.1f}%"""
    
    def demo_random_cards(self, cards_dir="/app/cards", num_cards=5):
        """Демонстрация на случайных картах"""
        print(f"\n🎯 Демонстрация распознавания на {num_cards} случайных картах\n")
        
        # Сбор всех изображений
        all_images = []
        for root, dirs, files in os.walk(cards_dir):
            for file in files:
                if file.endswith('.webp'):
                    all_images.append(os.path.join(root, file))
        
        if len(all_images) == 0:
            print("❌ Изображения не найдены в датасете")
            return
        
        # Случайный выбор
        random.seed(42)
        selected_images = random.sample(all_images, min(num_cards, len(all_images)))
        
        correct_predictions = 0
        
        for i, img_path in enumerate(selected_images, 1):
            print(f"\n{'='*60}")
            print(f"🔍 Тест {i}/{len(selected_images)}: {os.path.basename(img_path)}")
            print(f"{'='*60}")
            
            # Извлечение истинного класса
            true_class = self.extract_true_class(img_path)
            print(f"📋 Истинный класс: {true_class}")
            
            # Предсказание
            result = self.predict_card(img_path)
            if result:
                predictions = result['predictions']
                
                print(f"\n🤖 Предсказания модели:")
                for j, pred in enumerate(predictions, 1):
                    marker = "🎯" if j == 1 else "📊"
                    print(f"  {marker} {j}. {pred['class']} ({pred['percentage']:.1f}%)")
                    if j == 1:
                        print(f"      {self.format_prediction_text(pred).replace(chr(10), chr(10) + '      ')}")
                
                # Проверка правильности
                if predictions[0]['class'] == true_class:
                    print(f"\n✅ ПРАВИЛЬНО! Модель корректно распознала карту")
                    correct_predictions += 1
                else:
                    print(f"\n❌ ОШИБКА. Ожидалось: {true_class}")
            else:
                print("❌ Ошибка предсказания")
        
        # Итоговая статистика
        accuracy = (correct_predictions / len(selected_images)) * 100
        print(f"\n{'='*60}")
        print(f"📊 ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'='*60}")
        print(f"Правильных предсказаний: {correct_predictions}/{len(selected_images)}")
        print(f"Точность: {accuracy:.1f}%")
        
        if accuracy >= 90:
            print("🏆 Отличная точность!")
        elif accuracy >= 70:
            print("👍 Хорошая точность")
        else:
            print("⚠️ Требуется дополнительное обучение")
    
    def extract_true_class(self, img_path):
        """Извлечение истинного класса из пути к файлу"""
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
        
        return f"{set_name}_{card_number}_{card_type}"
    
    def predict_single_image(self, image_path):
        """Предсказание для одного изображения"""
        print(f"\n🔍 Анализ изображения: {image_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(image_path):
            print(f"❌ Файл не найден: {image_path}")
            return
        
        result = self.predict_card(image_path)
        if result:
            predictions = result['predictions']
            
            print(f"\n🤖 Результаты распознавания:")
            for i, pred in enumerate(predictions, 1):
                marker = "🎯" if i == 1 else "📊"
                print(f"\n{marker} Вариант {i}:")
                print(f"   Класс: {pred['class']}")
                print(f"   {self.format_prediction_text(pred).replace(chr(10), chr(10) + '   ')}")
            
            # Лучшее предсказание
            best = predictions[0]
            print(f"\n🏆 Лучшее предсказание: {best['class']} ({best['percentage']:.1f}%)")
        else:
            print("❌ Ошибка анализа изображения")

def main():
    """Основная функция"""
    print("🎮 Демонстрация модели распознавания карт")
    print("="*50)
    
    demo = CardRecognitionDemo()
    
    if demo.interpreter is None or len(demo.class_names) == 0:
        print("\n❌ Не удалось загрузить модель или классы")
        print("Убедитесь, что модель обучена: python train_model.py")
        return 1
    
    if len(sys.argv) > 1:
        # Анализ конкретного изображения
        image_path = sys.argv[1]
        demo.predict_single_image(image_path)
    else:
        # Демонстрация на случайных картах
        demo.demo_random_cards()
    
    print("\n✨ Демонстрация завершена!")
    return 0

if __name__ == "__main__":
    sys.exit(main())