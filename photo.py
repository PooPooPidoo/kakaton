import cv2
import numpy as np
from PIL import Image, ImageDraw


class MeterWater():
    def __init__(self, img):
        self.img = img
        self.digits_img = None
        self.digit_base_h = 24
        self.digit_base_w = 16
        self.result = None
        self.digits = None

    def loadImage(self, name_img):

        image = Image.open(name_img)  # Открываем изображение
        draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования    ДОБАВЛЕННЫЙ КОД
        width = image.size[0]  # Определяем ширину
        height = image.size[1]
        pix = image.load()

        for x in range(width):
            for y in range(height):
                r = pix[x, y][0]
                g = pix[x, y][1]
                b = pix[x, y][2]
                if r > 100:
                    r = 255
                    draw.point((x, y), (r, g, b))
                if g > 100:
                    g = 255
                    draw.point((x, y), (r, g, b))
                if b > 100:
                    b = 255
                    draw.point((x, y), (r, g, b))

        image.save("result.jpg", "JPEG")

        img = cv2.imread(name_img)   #в параметрах было name_img
        img_array = np.asarray(bytearray(img), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    def extractDigitsFromImage(self):
        img = self.img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        self.digits_img = img
        return True

    def splitDigits(self):

        # проверяем, если циферблат ещё не выделен, то делаем это
        if self.digits_img is None:
            if not self.extractDigitsFromImage():
                return False

        img = self.digits_img
        h, w, k = img.shape

        # разбиваем циферблат на 5 равных частей и обрабатываем каждую часть
        for i in range(1, 5):
            digit = img[0:h, (i - 1) * w / 5:i * w / 5]
            dh, dw, dk = digit.shape
            # не переводим в ч/б
            #digit_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
            #digit_bin = cv2.adaptiveThreshold(digit_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 0)


            # удаляем шум
            kernel = np.ones((2, 2), np.uint8)
            digit_bin = cv2.morphologyEx(digit_bin, cv2.MORPH_OPEN, kernel)

            # ищем контуры
            other, contours, hierarhy = cv2.findContours(digit_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # анализируем контуры
            biggest_contour = None
            biggest_contour_area = 0
            for cnt in contours:
                M = cv2.moments(cnt)

                # пропускаем контуры со слишком маленькой площадью
                if cv2.contourArea(cnt) < 30:
                    continue
                # пропускаем контуры со слишком маленьким периметром
                if cv2.arcLength(cnt, True) < 30:
                    continue

                # находим центр масс контура
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']

                # пропускаем контур, если центр масс находится где-то с краю
                if cx / dw < 0.3 or cx / dw > 0.7:
                    continue

                # находим наибольший контур
                if cv2.contourArea(cnt) > biggest_contour_area:
                    biggest_contour = cnt
                    biggest_contour_area = cv2.contourArea(cnt)
                    biggest_contour_cx = cx
                    biggest_contour_cy = cy

                # если не найдено ни одного подходящего контура, то помечаем цифру не распознанной
                if biggest_contour == None:
                    digit = self.dbDigit(i, digit_bin)
                    digit.markDigitForManualRecognize(use_for_training=False)
                    # mylogger.warn("Digit %d: no biggest contour found" % i)
                    continue

                # убираем всё, что лежит за пределами самого большого контура
                mask = np.zeros(digit_bin.shape, np.uint8)
                cv2.drawContours(mask, [biggest_contour], 0, 255, -1)
                digit_bin = cv2.bitwise_and(digit_bin, digit_bin, mask=mask)

                # задаем параметры описывающего прямоугольника
                rw = dw / 2.0
                rh = dh / 1.4

                # проверяем, чтобы прямоугольник не выходил за пределы изображения
                if biggest_contour_cy - rh / 2 < 0:
                    biggest_contour_cy = rh / 2
                if biggest_contour_cx - rw / 2 < 0:
                    biggest_contour_cx = rw / 2

                # вырезаем прямоугольник
                digit_bin = digit_bin[int(biggest_contour_cy - rh / 2):int(biggest_contour_cy + rh / 2),
                            int(biggest_contour_cx - rw / 2):int(biggest_contour_cx + rw / 2)]

                # изменяем размер на стандартный
                digit_bin = cv2.resize(digit_bin, (self.digit_base_w, self.digit_base_h))
                digit_bin = cv2.threshold(digit_bin, 128, 255, cv2.THRESH_BINARY)[1]

                # сохраняем в базу
                digit = self.dbDigit(i, digit_bin)

        return True

    def identifyDigits(self):

        # если число уже распознано, то ничего не делаем
        if self.result != '':
            return True

        # если цифры ещё не выделены
        if len(self.digits) == 0:
            # если изображение не задано, то ничего не получится
            if self.img == None:
                return False
            # выделяем цифры
            if not self.splitDigits():
                return False
            # утверждаем изменения в базу, которые сделаны при выделении цифр
            # sess.commit()

        # пытаемся распознать каждую цифру
        for digit in self.digits:
            digit.identifyDigit()

        # получаем текстовые значения цифр
        str_digits = map(str, self.digits)

        # если хотя бы одна цифра не распознана, то показание также не может быть распознано
        if '?' in str_digits:
            return False

        # склеиваем все цифры для получения числа
        self.result = ''.join(str_digits)
        return True

    def identifyDigit(self):

        # если цифра уже распознана, то ничего не делаем
        if self.result != '?':
            return True

        if not KNN.recognize(self):
            # если не удалось распознать цифру, то помечаем её для ручной обработки
            self.markDigitForManualRecognize()
            # если это 7-я цифра, то считаем её равной "0", так как это последняя цифра и не критичная, а часто бывает, что она не распознается
            if self.i == 7:
                self.result = 0
                return True
            return False
        else:
            self.use_for_training = True

        return True