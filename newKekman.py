from PIL import Image, ImageDraw


image = Image.open('image.jpg')  # Открываем изображение
draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования
width = image.size[0]  # Определяем ширину
height = image.size[1]
pix = image.load()

for x in range(width):
    for y in range(height):
       r = pix[x, y][0] #узнаём значение красного цвета пикселя
       g = pix[x, y][1] #зелёного
       b = pix[x, y][2] #синего
       sr = (r + g + b) // 3 #среднее значение
       draw.point((x, y), (sr, sr, sr))

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

