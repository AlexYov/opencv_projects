import cv2
import dlib

def draw_text_info():
    """Draw text information"""
    # указываем текст в видеопотоке

    # указываем координаты расположения текста
    menu_pos_1 = (10, 20)
    menu_pos_2 = (10, 40)

    # указываем параметры текста
    cv2.putText(frame, "Нажми 1, чтобы перезапустить трекер", menu_pos_1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255)) # функция putText() принимает параметры: изображение; сам текст, который будет на картинке; координаты расположения текста; какой шрифт; какой размер текста; цвет текста 
    if tracking_face:
        cv2.putText(frame, "Отслеживаю лицо", menu_pos_2, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "Ищем лицо для дальнейшего отслеживания его", menu_pos_2, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))


# захватываем кадры
capture = cv2.VideoCapture(0)

# переменная detector содержит функцию get_frontal_face_detector() для распознавания лица
detector = dlib.get_frontal_face_detector()

# запускаем корреляционный трекер, который отслеживает объект
tracker = dlib.correlation_tracker()

# переменаня tracking_face говорит отслеживаем ли объект или нет
tracking_face = False

while True:
    # берём кадры из видеопотока
    ret, frame = capture.read()

    # вставляем текст в видеопоток
    draw_text_info()

    if tracking_face == False: 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # находим лицо на кадре
        rects = detector(gray, 0)
 
        # проверяем обнаружили ли лицо или нет
        if len(rects) > 0:
  
            #передаём трекеру первый кадр из видеопотока. трекер анализиуерт полученное изображение и на основании его, отслеживает объект
            tracker.start_track(frame, rects[0])
            tracking_face = True

    if tracking_face == True:
        # обновляем трекер, чтобы в каждом новом кадре отслеживал объект
        tracker.update(frame)
        # получем расположение объекта трекера
        pos = tracker.get_position()
        # рисуем квадрат, вокруг объекта, который нашли
        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (255, 255, 255), 1)
        
    
    button = 0xFF & cv2.waitKey(1) # 0xFF (0b11111111)
    
    # нажимаем 1, чтобы заново найти объект в видеопотоке
    if button == ord("1"):
        tracking_face = False

    # нажимаем Esc, чтобы выйти
    if button == 27:
        break
    
    # показываем результат
    cv2.imshow("Detecting and tracking face", frame)

# останавливаем программу и закрываем все окна
capture.release()
cv2.destroyAllWindows()