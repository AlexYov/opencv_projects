import os # модуль предоставляет функции для работы с операционной системой
import dlib # библиотека машинного обучения
import argparse # модуль для обработки аргументов командной строки
import glob # находит все пути в операционной системе, совпадающие с заданным шаблоном

parser = argparse.ArgumentParser() # объект ArgumentParser() для парсинга аргументов

parser.add_argument('--path_photos', dest = 'path_photos', help = 'Путь к папке с изображениями/фотографиями') # обязательный аргумент, который принимает путь к папке, где лежат изображения, на которых нужно найти объект

parser.add_argument('--path_train', dest = 'path_train', help = 'Путь к базе с данными для тренировки системы') # обязательный аргумент, который принимает путь к файлу для тренировки системы распознавания объектов

args = parser.parse_args() # содержатся все аргументы, которые были переданы скрипту

options = dlib.simple_object_detector_training_options() # функция simple_object_detector_training_options() содержит опции для тренировки системы

options.add_left_right_image_flips = True

options.C = 5

options.num_threads = 4 # передаём количество ядер компьютера, которые можно использовать для обучения. чем больше, тем быстрее обучается система
 
dlib.train_simple_object_detector(args.path_train, "detector.svm", options) # эта функция обучает систему. передаются параметры: путь к файлу для тренировки, имя выходного файла, опции тренировки

detector = dlib.simple_object_detector("detector.svm") # передаём в переменную detector метод simple_object_detector(), обнаруживающий объекты на основе гистограммы направленных градиентов (Histogram of oriented gradients - HOG)

window = dlib.image_window() # объявляем графическое окно для отображения изображений

for file in glob.glob(os.path.join(args.path_photos,'*')): # перебираем каждое изображение

    image = dlib.load_rgb_image(file) # передаём изображение и получаем массив из RGB значений
    
    dets = detector(image) # в каждом изображении система ищет необходимый объект
    
    window.clear_overlay() # убираем вверхний слой на изображении, на котором ранее были прямоугольники
    
    window.set_image(image) # показываем итоговое изображение с найденным объектом
    
    window.add_overlay(dets) # на каждом изображение обводим прямоугольником найденный объект (накладывается новый слой на изображение)
    
    dlib.hit_enter_to_continue() # при нажатии на любую клавишу, показывает итоговые изображения по очереди