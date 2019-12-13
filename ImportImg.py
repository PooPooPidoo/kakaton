import cv2
import numpy as np


def drawContour(img: str, contour: str):
    cnt_image = cv2.drawContours(img, contour, -1, [0, 0, 250], 1, cv2.LINE_AA)
    cv2.imshow('1', cnt_image)
    cv2.waitKey(0)


def getGameFieldImg(image_file: str):
    origImg = cv2.imread(image_file)
    defaultImg = origImg[60:750, 345:1020]
    cv2.imshow('1', defaultImg)
    cv2.waitKey()
    grayImg = cv2.cvtColor(defaultImg, cv2.COLOR_RGB2GRAY)
    ret, threshold1 = cv2.threshold(grayImg, 251, 253, cv2.THRESH_BINARY)
    # img_erode = cv2.erode(threshold1, np.ones((1, 2), np.uint16), iterations=1)
    # img_erode = cv2.bilateralFilter(img_erode, 1, 2, 2)
    letters = letters_extract(defaultImg, grayImg, threshold1)


def letters_extract(originalImage, grayImg, threshold, out_size=28):
    # Get contours
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    drawContour(originalImage, contours)
    output = originalImage.copy()
    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == -1:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = grayImg[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    cv2.imshow("Output", output)
    cv2.imshow("thresh", threshold)
    cv2.imshow("0", letters[0][2])
    cv2.imshow("1", letters[1][2])
    cv2.imshow("2", letters[2][2])
    cv2.imshow("3", letters[3][2])
    cv2.imshow("4", letters[4][2])

    # cv2.imshow('erode', img_erode)
    cv2.waitKey(0)
    return letters


def getEnemyBoardStats(img: str):
    defaultImg = cv2.imread(img)
    statsEnemBoard = defaultImg[310:329, 345:1020]
    resized = cv2.resize(statsEnemBoard, (int(statsEnemBoard.shape[1] * 2), int(statsEnemBoard.shape[0]) * 2))
    cv2.imshow('1', resized)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    # ret, thresholdUp = cv2.threshold(grayUp, 10, 20, cv2.THRESH_BINARY)
    ret, threshold = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    cv2.imshow('Up', threshold)
    cv2.waitKey(0)
    letters_extract(resized, gray, threshold)


def getDownerBoardStats(img: str):
    defaultImg = cv2.imread(img)
    statsDown = defaultImg[443:461, 345:1020]
    resized = cv2.resize(statsDown, (int(statsDown.shape[1] * 2), int(statsDown.shape[0]) * 2))
    cv2.imshow('1', resized)
    grayDown = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    ret, thresholdDown = cv2.threshold(grayDown, 10, 30, cv2.THRESH_BINARY_INV)
    cv2.imshow('Down', thresholdDown)
    cv2.waitKey(0)
    letters_extract(resized, grayDown, thresholdDown)


# getDownerBoardStats("2.png")
# getEnemyBoardStats("2.png")
getGameFieldImg("1.png")
# img_erode = cv2.erode(threshold1, np.ones((1, 2), np.uint8), iterations=1)
# ret, thresholdUp = cv2.threshold(grayUp, 10, 20, cv2.THRESH_BINARY)
