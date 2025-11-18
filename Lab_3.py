import numpy as np
import math
import random
from PIL import Image, ImageOps
img_mat = np.zeros((2000,2000,3), dtype=np.uint8)
z_buffer = np.full((2000, 2000), float('inf'))
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
"""
def lamda(x0, y0, x1, y1, x2, y2, x, y):
    if ((1-((x-x2)*(y1-y2)-(x1-x2)*(y-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2))-((x0-x2)*(y-y2)-(x-x2)*(y0-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2)))>=0
            and (((x-x2)*(y1-y2)-(x1-x2)*(y-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2)))>=0
            and (((x0-x2)*(y-y2)-(x-x2)*(y0-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2)))>=0):
        return True
    return False;
"""
def rot_matrix_X(angle):
    return np.array([[1,0,0],[0,math.cos(angle),math.sin(angle)],[0,-math.sin(angle),math.cos(angle)]])

def rot_matrix_Y(angle):
    return np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])

def rot_matrix_Z(angle):
    return np.array([
        [math.cos(angle), math.sin(angle), 0],
        [-math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])
def create_coord(alpha, beta, gamma):
    R_x = rot_matrix_X(alpha)
    R_y = rot_matrix_Y(beta)
    R_z = rot_matrix_Z(gamma)
    return rot_matrix_Z(gamma) @ rot_matrix_Y(beta) @ rot_matrix_X(alpha)


def finalfunction(vertex, alpha, beta, gamma, t):
    R = create_coord(alpha, beta, gamma)

    transformed_vertex = R @ np.array(vertex) + np.array(t)

    return transformed_vertex.tolist()

def lamda1(x0, y0, x1, y1, x2, y2, x, y):
    denominator = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    if abs(denominator) < 1e-10:
        return 0
    return ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2))/denominator
def lamda0(x0, y0, x1, y1, x2, y2, x, y):
    denominator = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    if abs(denominator) < 1e-10:
        return 0
    return ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denominator
def lamda2(x0, y0, x1, y1, x2, y2, x, y):return 1-lamda0(x0,y0, x1, y1, x2, y2, x, y)-lamda1(x0, y0, x1, y1, x2, y2, x, y)

def normal(v1, v2, v3):
    norm = []
    v = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
    u = [v2[0] - v3[0], v2[1] - v3[1], v2[2] - v3[2]]

    nx = u[1] * v[2] - u[2] * v[1]
    ny = u[2] * v[0] - u[0] * v[2]
    nz = u[0] * v[1] - u[1] * v[0]
    length = math.sqrt(nx * nx + ny * ny + nz * nz)


    if length > 0:
        nx = nx / length
        ny = ny / length
        nz = nz / length

    return [nx, ny, nz]
def proverka(v1,v2,v3):
    denominator =np.linalg.norm(normal(v1,v2,v3))
    if abs(denominator) < 1e-10:
        return 0
    return np.dot(normal(v1,v2,v3),[0,0,1])/denominator
def otrisovka(image, x0, y0, x1, y1,x2,y2,z0,z1,z2,color):

    x0_s = (1000 * x0)/z0 + 1000
    y0_s = (1000 * y0)/z0 + 1000
    x1_s = (1000* x1)/z1 + 1000
    y1_s = (1000 * y1)/z1 + 1000
    x2_s = (1000 * x2)/z2 + 1000
    y2_s = (1000 * y2)/z2 + 1000
    xmin = max(0, min(int(x0_s), int(x1_s), int(x2_s)))
    xmax = min(image.shape[1] - 1, max(int(x0_s), int(x1_s), int(x2_s)))
    ymin = max(0, min(int(y0_s), int(y1_s), int(y2_s)))
    ymax = min(image.shape[0] - 1, max(int(y0_s), int(y1_s), int(y2_s)))

    if ((proverka([x0, y0, z0], [x1, y1, z1], [x2, y2, z2]) < 0)):
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                if ( lamda0(x0_s,y0_s,x1_s,y1_s,x2_s,y2_s,x,y)>=0 and lamda1(x0_s,y0_s,x1_s,y1_s,x2_s,y2_s,x,y)>=0 and
                        lamda2(x0_s, y0_s, x1_s, y1_s, x2_s, y2_s, x, y) >= 0):
                    m=lamda0(x0_s, y0_s, x1_s, y1_s, x2_s, y2_s, x, y)*z0+lamda1(x0_s, y0_s, x1_s, y1_s, x2_s, y2_s, x, y)*z1+lamda2(x0_s, y0_s, x1_s, y1_s, x2_s, y2_s, x, y)*z2
                    if (m<z_buffer[y,x]):
                        image[y, x] = color
                        z_buffer[y,x] = m


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

alpha, beta, gamma = 0.0, 0.0, 0.0
t = [0.0, -0.03, 0.1]

for i in range(len(a)):
    a[i] = finalfunction(a[i], alpha, beta, gamma, t)

n=len(a)


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
    v0_orig = a[c[k][0] - 1]
    v1_orig = a[c[k][1] - 1]
    v2_orig = a[c[k][2] - 1]
    x0 = (v0_orig[0] )
    y0 = (v0_orig[1] )
    x1 = (v1_orig[0] )
    y1 = (v1_orig[1] )
    x2 = (v2_orig[0] )
    y2 = (v2_orig[1] )
    z0 = v0_orig[2]
    z1 = v1_orig[2]
    z2 = v2_orig[2]
    color = [-255*(proverka([x0,y0,z0],[x1,y1,z1],[x2,y2,z2])),0,0]

    otrisovka(img_mat, x0, y0, x1, y1, x2, y2,z0,z1,z2, color)

img=Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img4.png')