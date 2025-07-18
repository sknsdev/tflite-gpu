#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для управления проектом распознавания карт
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd, cwd=None):
    """Выполнение команды в терминале"""
    print(f"Выполнение: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка выполнения команды: {e}")
        return False

def check_docker():
    """Проверка доступности Docker"""
    try:
        subprocess.run(["docker", "--version"], check=True, 
                      capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Ошибка: Docker не найден или не запущен")
        return False

def check_nvidia_docker():
    """Проверка поддержки NVIDIA Docker"""
    try:
        result = subprocess.run(["docker", "run", "--rm", "--gpus", "all", 
                               "nvidia/cuda:11.0-base", "nvidia-smi"], 
                              check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        print("Предупреждение: NVIDIA Docker может быть недоступен")
        return False

def build_image():
    """Сборка Docker образа"""
    print("Сборка Docker образа...")
    return run_command("docker build -t card-recognition .")

def check_gpu():
    """Проверка GPU в контейнере"""
    print("Проверка GPU...")
    cmd = "docker run --rm --gpus all -v %cd%:/app card-recognition python check_gpu.py"
    return run_command(cmd)

def train_model():
    """Запуск обучения модели"""
    print("Запуск обучения модели...")
    cmd = "docker run -it --gpus all -v %cd%:/app --name card-training card-recognition python train_model.py"
    return run_command(cmd)

def test_model(image_path=None):
    """Тестирование модели"""
    print("Тестирование модели...")
    if image_path:
        cmd = f"docker run --rm --gpus all -v %cd%:/app card-recognition python test_model.py {image_path}"
    else:
        cmd = "docker run --rm --gpus all -v %cd%:/app card-recognition python test_model.py"
    return run_command(cmd)

def demo_model(image_path=None):
    """Демонстрация модели"""
    print("Запуск демонстрации модели...")
    if image_path:
        cmd = f"docker run --rm --gpus all -v %cd%:/app card-recognition python demo.py {image_path}"
    else:
        cmd = "docker run --rm --gpus all -v %cd%:/app card-recognition python demo.py"
    return run_command(cmd)

def clean_containers():
    """Очистка контейнеров"""
    print("Очистка контейнеров...")
    run_command("docker container rm -f card-training card-recognition-training card-recognition-test")
    run_command("docker container prune -f")

def show_status():
    """Показать статус проекта"""
    print("=== Статус проекта ===")
    
    # Проверка файлов
    required_files = [
        "train_model.py",
        "test_model.py", 
        "check_gpu.py",
        "requirements.txt",
        "Dockerfile"
    ]
    
    print("\nФайлы проекта:")
    for file in required_files:
        exists = "✓" if os.path.exists(file) else "✗"
        print(f"  {exists} {file}")
    
    # Проверка датасета
    print("\nДатасет:")
    cards_dir = "cards"
    if os.path.exists(cards_dir):
        total_images = 0
        sets = []
        for set_name in os.listdir(cards_dir):
            set_path = os.path.join(cards_dir, set_name)
            if os.path.isdir(set_path):
                sets.append(set_name)
                for card_folder in os.listdir(set_path):
                    card_path = os.path.join(set_path, card_folder)
                    if os.path.isdir(card_path):
                        images = len([f for f in os.listdir(card_path) if f.endswith('.webp')])
                        total_images += images
        
        print(f"  ✓ Найдено сетов: {len(sets)}")
        print(f"  ✓ Всего изображений: {total_images}")
        print(f"  ✓ Сеты: {', '.join(sets)}")
    else:
        print("  ✗ Папка cards не найдена")
    
    # Проверка модели
    print("\nОбученная модель:")
    model_exists = "✓" if os.path.exists("card_recognition_model.tflite") else "✗"
    classes_exists = "✓" if os.path.exists("class_names.json") else "✗"
    print(f"  {model_exists} card_recognition_model.tflite")
    print(f"  {classes_exists} class_names.json")

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Управление проектом распознавания карт")
    parser.add_argument("command", choices=[
        "build", "check", "train", "test", "demo", "clean", "status", "all"
    ], help="Команда для выполнения")
    parser.add_argument("--image", help="Путь к изображению для тестирования")
    
    args = parser.parse_args()
    
    # Проверка Docker
    if not check_docker():
        return 1
    
    if args.command == "build":
        if not build_image():
            return 1
            
    elif args.command == "check":
        if not check_gpu():
            return 1
            
    elif args.command == "train":
        clean_containers()
        if not train_model():
            return 1
            
    elif args.command == "test":
        if not test_model(args.image):
            return 1
            
    elif args.command == "demo":
        if not demo_model(args.image):
            return 1
            
    elif args.command == "clean":
        clean_containers()
        
    elif args.command == "status":
        show_status()
        
    elif args.command == "all":
        print("Полный цикл: сборка -> проверка -> обучение -> тестирование -> демонстрация")
        if not build_image():
            return 1
        if not check_gpu():
            return 1
        clean_containers()
        if not train_model():
            return 1
        if not test_model():
            return 1
        if not demo_model():
            return 1
    
    print("\n✓ Команда выполнена успешно!")
    return 0

if __name__ == "__main__":
    sys.exit(main())