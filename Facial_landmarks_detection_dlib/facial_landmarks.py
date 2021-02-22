import cv2
import dlib
import numpy as np

# задаём черты лица
JAWLINE_POINTS = list(range(0, 17)) # линия подбородка
RIGHT_EYEBROW_POINTS = list(range(17, 22)) # права бровь
LEFT_EYEBROW_POINTS = list(range(22, 27)) # левая бровь
NOSE_BRIDGE_POINTS = list(range(27, 31)) # линия носа
LOWER_NOSE_POINTS = list(range(31, 36)) # нижняя часть носа
RIGHT_EYE_POINTS = list(range(36, 42)) # правый глаз
LEFT_EYE_POINTS = list(range(42, 48)) # левый глаз
MOUTH_OUTLINE_POINTS = list(range(48, 60)) # очертание наружной части рта
MOUTH_INNER_POINTS = list(range(60, 68)) # очертания внутреней части рта
ALL_POINTS = list(range(0, 68)) # все точки


def draw_shape_lines_all(np_shape, image):
    """Draws the shape using lines to connect between different parts of the face(e.g. nose, eyes, ...)"""
    # рисуем фигуру, используя линии, чтобы соединить разные части лица
    draw_shape_lines_range(np_shape, image, JAWLINE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, LEFT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, NOSE_BRIDGE_POINTS)
    draw_shape_lines_range(np_shape, image, LOWER_NOSE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, LEFT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_OUTLINE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_INNER_POINTS, True)


def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    """Draws the shape using lines to connect the different points"""
    # рисуем фигуру, используя линии, чтобы чтобы соединить разные точки
    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (206, 255, 255), thickness=1, lineType=cv2.LMEDS)


def draw_shape_points_pos_range(np_shape, image, points):
    """Draws the shape using points and position for every landmark filtering by points parameter"""
    # Рисует фигуру с использованием точек и положения для каждый черты лица, фильтруя по параметру точек
    np_shape_display = np_shape[points]
    draw_shape_points_pos(np_shape_display, image)


def draw_shape_points_pos(np_shape, image):
    """Draws the shape using points and position for every landmark"""
    # рисуем фигуру, используя точки и расположение для каждой черты лица
    for idx, (x, y) in enumerate(np_shape):
        # рисуем позиции для каждой обнаруженной черты лица
        cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (224,255,255))

        # рисуем точку на каждой позиции черты дица
        cv2.circle(image, (x, y), 2, (255,99,71), -1)


def draw_shape_points_range(np_shape, image, points):
    """Draws the shape using points for every landmark filtering by points parameter"""
    # рисуем фигуру, используя точки для каждой черты лица, фильтруя по параметру точек
    np_shape_display = np_shape[points]
    draw_shape_points(np_shape_display, image)


def draw_shape_points(np_shape, image):
    """Draws the shape using points for every landmark"""
    # рисуем фигуру, используя точки для каждой черты лица
    
    # рисуем точку на каждой позиции черты лица
    for (x, y) in np_shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def shape_to_np(dlib_shape, dtype="int"):
    """Converts dlib shape object to numpy array"""
    # Преобразует объект формы dlib в массив numpy

    # Инициализировать список координат (x, y)
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

    # зацикливаем все черты лица и конвертируем их в кортеж с координатами (x,y)
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

    # возвращаем список координат (x,y)
    return coordinates

# определители форм
p = "shape_predictor_68_face_landmarks.dat"
#p = "shape_predictor_5_face_landmarks.dat"

# запускаем фронтальный распознаватель лица и определитель форм
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# запускаем видеопоток
video_capture = cv2.VideoCapture(0)

while True:

    # получаем каждый кадр из видеопотока
    bool_result, frame = video_capture.read()

    # преобразуем цветной кадр в нецветной
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # распознаём лицо
    rects = detector(gray, 0)

    # находим черты лица для каждого распознанного лица
    for (i, rect) in enumerate(rects):
        # рисуем рамку вокруг лица
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255,228,225), 1)

        # получаем фигуру, используя определитель
        shape = predictor(gray, rect)

        # преобразуем фигуру в масси numpy
        array_shape = shape_to_np(shape)

        # рисуем все линии, соединяя разные части лица
        draw_shape_lines_all(array_shape, frame)

        # рисуем линию подбородка
        #draw_shape_lines_range(array_shape, frame, JAWLINE_POINTS)
        
        # рисуем глаза
        #draw_shape_lines_range(array_shape, frame, RIGHT_EYE_POINTS, True)
        #draw_shape_lines_range(array_shape, frame, LEFT_EYE_POINTS, True)
        
        # рисуем брови
        #draw_shape_lines_range(array_shape, frame, RIGHT_EYEBROW_POINTS)
        #draw_shape_lines_range(array_shape, frame, LEFT_EYEBROW_POINTS)
        
        # рисуем нос
        #draw_shape_lines_range(array_shape, frame, NOSE_BRIDGE_POINTS)
        #draw_shape_lines_range(array_shape, frame, LOWER_NOSE_POINTS)        
        
        # рисуем рот
        #draw_shape_lines_range(array_shape, frame, MOUTH_OUTLINE_POINTS, True)
        #draw_shape_lines_range(array_shape, frame, MOUTH_INNER_POINTS, True)          

        # рисуем все точки и их положение
        #draw_shape_points_pos(array_shape, frame)
        # также можно использовать и такой метод
        #draw_shape_points_pos_range(array_shape, frame, ALL_POINTS)
        
        # рисуем левую и правую бровь
        #draw_shape_points_pos_range(array_shape, frame, RIGHT_EYEBROW_POINTS + LEFT_EYEBROW_POINTS)
        
        # рисуем нос
        #draw_shape_points_pos_range(array_shape, frame, NOSE_BRIDGE_POINTS + LOWER_NOSE_POINTS)
        
        # рисуем глаза
        #draw_shape_points_pos_range(array_shape, frame, RIGHT_EYE_POINTS + LEFT_EYE_POINTS)
        
        # рисуем рот
        #draw_shape_points_pos_range(array_shape, frame, MOUTH_OUTLINE_POINTS + MOUTH_INNER_POINTS)
        
        #
        #draw_shape_points_pos_range(array_shape, frame, JAWLINE_POINTS)

        # рисуем все точки фигуры
        #draw_shape_points(array_shape, frame)

    # показываем результат
    cv2.imshow("Facial landmarks detection", frame)

    # выход по кнопке Esc
    if cv2.waitKey(1) == 27:
        break


#  закрываем все окна и полностью всю программу
video_capture.release()
cv2.destroyAllWindows()