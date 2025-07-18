#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки доступности GPU и настройки TensorFlow
"""

import tensorflow as tf
import os

def check_gpu_availability():
    """Проверка доступности GPU"""
    print("=== Проверка GPU ===")
    print(f"TensorFlow версия: {tf.__version__}")
    print(f"Встроенная поддержка CUDA: {tf.test.is_built_with_cuda()}")
    
    # Список физических устройств
    physical_devices = tf.config.list_physical_devices()
    print(f"\nВсе физические устройства: {len(physical_devices)}")
    for device in physical_devices:
        print(f"  - {device}")
    
    # GPU устройства
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"\nGPU устройства: {len(gpu_devices)}")
    for i, gpu in enumerate(gpu_devices):
        print(f"  GPU {i}: {gpu}")
        
        # Детали GPU
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"    Детали: {gpu_details}")
        except:
            print("    Детали недоступны")
    
    # Проверка доступности CUDA
    print(f"\nCUDA доступна: {tf.test.is_gpu_available()}")
    
    if len(gpu_devices) > 0:
        print("\n✓ GPU найден и готов к использованию!")
        
        # Настройка роста памяти
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ Настроен рост памяти GPU")
        except Exception as e:
            print(f"⚠ Ошибка настройки памяти GPU: {e}")
            
    else:
        print("\n⚠ GPU не найден, будет использоваться CPU")
    
    return len(gpu_devices) > 0

def test_gpu_computation():
    """Тест вычислений на GPU"""
    print("\n=== Тест вычислений ===")
    
    try:
        # Создание тензоров
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            # Матричное умножение
            c = tf.matmul(a, b)
            
        device_name = c.device
        print(f"Вычисления выполнены на: {device_name}")
        print(f"Результат: тензор размером {c.shape}")
        print("✓ Тест вычислений прошел успешно!")
        
    except Exception as e:
        print(f"✗ Ошибка при тестировании: {e}")

def check_dataset():
    """Проверка датасета"""
    print("\n=== Проверка датасета ===")
    
    cards_dir = "/app/cards"
    if not os.path.exists(cards_dir):
        print(f"⚠ Папка датасета не найдена: {cards_dir}")
        return False
    
    total_images = 0
    sets_count = 0
    cards_count = 0
    
    for set_name in os.listdir(cards_dir):
        set_path = os.path.join(cards_dir, set_name)
        if not os.path.isdir(set_path):
            continue
            
        sets_count += 1
        set_images = 0
        set_cards = 0
        
        for card_folder in os.listdir(set_path):
            card_path = os.path.join(set_path, card_folder)
            if not os.path.isdir(card_path):
                continue
                
            set_cards += 1
            cards_count += 1
            
            # Подсчет изображений в папке карты
            card_images = len([f for f in os.listdir(card_path) if f.endswith('.webp')])
            set_images += card_images
            total_images += card_images
        
        print(f"Сет '{set_name}': {set_cards} карт, {set_images} изображений")
    
    print(f"\nИтого:")
    print(f"  Сеты: {sets_count}")
    print(f"  Карты: {cards_count}")
    print(f"  Изображения: {total_images}")
    
    if total_images > 0:
        print("✓ Датасет найден и готов к использованию!")
        return True
    else:
        print("⚠ Изображения в датасете не найдены")
        return False

def main():
    """Основная функция"""
    print("Проверка окружения для обучения модели распознавания карт\n")
    
    # Проверка GPU
    gpu_available = check_gpu_availability()
    
    # Тест вычислений
    test_gpu_computation()
    
    # Проверка датасета
    dataset_ready = check_dataset()
    
    print("\n=== Итоговый статус ===")
    print(f"GPU доступен: {'✓' if gpu_available else '✗'}")
    print(f"Датасет готов: {'✓' if dataset_ready else '✗'}")
    
    if gpu_available and dataset_ready:
        print("\n🚀 Все готово для обучения модели!")
        print("Запустите: python train_model.py")
    else:
        print("\n⚠ Требуется настройка окружения")
        if not gpu_available:
            print("  - Проверьте установку CUDA и драйверов GPU")
        if not dataset_ready:
            print("  - Убедитесь, что папка cards содержит изображения")

if __name__ == "__main__":
    main()