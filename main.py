import numpy as np
import math
from PIL import Image, ImageOps
img_mat = np.zeros((2000,2000,3), dtype=np.uint8)
""""
for i in range(600):
    for j in range (800):
        img_mat[i,j]=[0,0,(i+j)%256]


def draw_line(image, x0, y0, x1, y1, color):
    count = 100
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def draw_line2(image, x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def draw_line3(image, x0, y0, x1, y1,color):
    xchange=False

    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def draw_line4(image, x0, y0, x1, y1,color):
    xchange = False

    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2.0 * (x1-x0) * abs(y1 - y0) / (x1 - x0)
    derror = 0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 2.0 * (x1-x0) * 0.5):
            derror -= 2.0 * (x1-x0) * 1.0
            y += y_update

"""

def draw_line5(image, x0, y0, x1, y1,color):
    xchange = False

    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > (x1-x0)):
            derror -= 2 * (x1-x0)
            y += y_update


"""""
for k in range (13):
    x0, y0 = 100, 100
    x1 = int( 100 + 95*math.cos(2*math.pi/13*k))
    y1 = int(100 + 95*math.sin(2*math.pi/13*k))
    #draw_line(img_mat, x0, y0, x1, y1, (255, 255, 255))
    #draw_line2(img_mat, x0, y0, x1, y1, (255,255,255))
    #draw_line3(img_mat, x0, y0, x1, y1, (255, 255, 255))
    #draw_line4(img_mat, x0, y0, x1, y1, (255, 255, 255))
    #draw_line5(img_mat, x0, y0, x1, y1, (255, 255, 255))
"""


file=open('model_1.obj')
a=[]
for i in file:
    word=i.split()

    if (word[0]=="v"):
        b=[]
        b.append(float (word[1]))
        b.append(float (word[2]))
        b.append(float (word[3]))
        a.append(b)
print(a)
n=len(a)
for i in range(n-1):
    img_mat[int(a[i][1]*10000+1000),int(a[i][0]*10000+1000)]=[255,255,255]

file.seek(0)
c=[]
for i in file:
    word=i.split()

    if (word[0]=="f"):
        x = int(word[1].split('/')[0])
        y = int(word[2].split('/')[0])
        z = int(word[3].split('/')[0])
        c.append([x,y,z])
for k in range(len(c)):
    x0 =int(a[c[k][0]- 1][0] *10000+1000)
    y0 = int(a[c[k][0] - 1][1] * 10000 + 1000)
    x1 = int(a[c[k][1] - 1][0] * 10000 + 1000)
    y1 =int(a[c[k][1] - 1][1] *10000 + 1000)
    x2 = int(a[c[k][2] - 1][0] * 10000 + 1000)
    y2 = int(a[c[k][2] - 1][1] * 10000 + 1000)
    draw_line5(img_mat, x0, y0, x1, y1, (255, 255, 255))
    draw_line5(img_mat, x1, y1, x2, y2, (255, 255, 255))
    draw_line5(img_mat, x2, y2, x0, y0, (255, 255, 255))


img=Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')