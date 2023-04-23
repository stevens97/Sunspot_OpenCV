# ---------------------------------
# Import Libraries
# ---------------------------------

import cv2  # For image processing
import numpy as np  # For scientific computing
from scipy.ndimage.measurements import label
import math


# Set path of sample image
# ---------------------------------
path = r'Sample.jpg'

# Read image with OpenCV2
# ---------------------------------
image = cv2.imread(path)
pixels = np.asarray(image)
y_len = len(pixels)
x_len = len(pixels[0])

# Convert image to binary colours to more easily detect sunspots
# ----------------------------------------------------------------
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold = cv2.adaptiveThreshold(grayscale, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 131,
                                  15)

thresh_pixels = np.asarray(threshold)

# Define potential sunspot groups
# ----------------------------------------------------------------
structure = np.ones((3, 3), dtype=np.int)
sunspot_Group, Number_Of_Potential_Sunspots = label(thresh_pixels,
                                                    structure)
# Calculate sunspot group sizes
# ----------------------------------------------------------------
sunspot_Group_Size = [(sunspot_Group == label).sum() for label in
                      range(Number_Of_Potential_Sunspots + 1)]

# Find center-point of solar disk
# ---------------------------------
counter = 1
sun_Edge_Left_x = x_len
sun_Edge_Left_y = 0
sun_Edge_Right_x = 0
sun_Edge_Right_y = 0
sun_Edge_Top_x = 0
sun_Edge_Top_y = y_len
sun_Edge_Bottom_x = 0
sun_Edge_Bottom_y = 0
for y in range(y_len):
    for x in range(x_len):
        if sunspot_Group[y][x] == counter:
            if x < sun_Edge_Left_x and sunspot_Group[y][x + 1] == counter:
                sun_Edge_Left_x = x
                sun_Edge_Left_y = y
            if x > sun_Edge_Right_x and sunspot_Group[y][x - 1] == counter:
                sun_Edge_Right_x = x
                sun_Edge_Right_y = y
            if y > sun_Edge_Bottom_y and sunspot_Group[y - 1][x] == counter:
                sun_Edge_Bottom_y = y
                sun_Edge_Bottom_x = x
            if y < sun_Edge_Top_y and sunspot_Group[y + 1][x] == counter:
                sun_Edge_Top_y = y
                sun_Edge_Top_x = x

# Find edges of solar disk
# ---------------------------------
flag = False
within_Sun = False
x = sun_Edge_Left_x
y = sun_Edge_Left_y
while flag is False:
    x = x + 1
    within_Sun = True
    if sunspot_Group[y][x] != 1:
        i = x
        while i <= x + 10:
            i = i + 1
            if sunspot_Group[y][i] == 1:
                within_Sun = False
        if within_Sun is True:
            sun_Inner_Edge_Left_x = x
            sun_Inner_Edge_Left_y = y
            flag = True

flag = False
within_Sun = False
x = sun_Edge_Right_x
y = sun_Edge_Right_y
while flag is False:
    x = x - 1
    within_Sun = True
    if sunspot_Group[y][x] != 1:
        i = x
        while i >= x - 10:
            i = i - 1
            if sunspot_Group[y][i] == 1:
                within_Sun = False
        if within_Sun is True:
            sun_Inner_Edge_Right_x = x
            sun_Inner_Edge_Right_y = y
            flag = True

flag = False
within_Sun = False
x = sun_Edge_Top_x
y = sun_Edge_Top_y
while flag is False:
    y = y + 1
    within_Sun = True
    if sunspot_Group[y][x] != 1:
        j = y
        while j <= y + 10:
            j = j + 1
            if sunspot_Group[x][j] == 1:
                within_Sun = False
        if within_Sun is True:
            sun_Inner_Edge_Top_x = x
            sun_Inner_Edge_Top_y = y
            flag = True

flag = False
within_Sun = False
x = sun_Edge_Bottom_x
y = sun_Edge_Bottom_y
while flag is False:
    y = y - 1
    within_Sun = True
    if sunspot_Group[y][x] != 1:
        j = y
        while j >= y - 10:
            j = j - 1
            if sunspot_Group[x][j] == 1:
                within_Sun = False
        if within_Sun is True:
            sun_Inner_Edge_Bottom_x = x
            sun_Inner_Edge_Bottom_y = y
            flag = True

x_Midpoint = sun_Inner_Edge_Left_x + (
        sun_Inner_Edge_Right_x - sun_Inner_Edge_Left_x) / 2
y_Midpoint = sun_Inner_Edge_Top_y + (
        sun_Inner_Edge_Bottom_y - sun_Inner_Edge_Top_y) / 2

r = (sun_Inner_Edge_Right_x - sun_Inner_Edge_Left_x) / 2.0


# Create solar disk
# ---------------------------------

def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def make_circle(tiles, cx, cy, r):
    for x in range(cx - r, cx + r):
        for y in range(cy - r, cy + r):
            if dist(cx, cy, x, y) <= r:
                tiles[x][y] = True

    return tiles

# Set tiles of solar disk
# ---------------------------------

tiles = [[0 for _ in range(y_len)] for _ in range(x_len)]
sun = make_circle(tiles, int(x_Midpoint), int(y_Midpoint), int(r))

sunspot_Detection_Minimum_Tolerance = 0.005

sunspot_Minimum_Size = sunspot_Detection_Minimum_Tolerance * x_len

sunspot_x_Centre = [0] * 100
sunspot_y_Centre = [0] * 100
sunspot_Size = [0] * 100

counter = 2
sunspot_Number = 0

# -------------------------------------------------------------------
# Detect coordinates of all potential sunspots on the solar disk
# -------------------------------------------------------------------

print('#-------------------------------------------------------------------')
print('Detecting coordinates of potential Sunspots...')
print('#-------------------------------------------------------------------')

while counter <= Number_Of_Potential_Sunspots:
    x_Total = 0
    y_Total = 0
    number_Of_Pixels = 0
    if sunspot_Minimum_Size < np.sqrt(sunspot_Group_Size[counter]):
        for y in range(y_len):
            for x in range(x_len):
                if sunspot_Group[y][x] == counter:
                    number_Of_Pixels = number_Of_Pixels + 1
                    x_Total = x_Total + x
                    y_Total = y_Total + y
        sunspot_x_Centre[sunspot_Number] = round(x_Total / number_Of_Pixels)
        sunspot_y_Centre[sunspot_Number] = round(y_Total / number_Of_Pixels)
        sunspot_Size[sunspot_Number] = number_Of_Pixels

        if sun[sunspot_x_Centre[sunspot_Number]][sunspot_y_Centre[sunspot_Number]] == True:
            message = "Sunspot #{} identified with pixel coordinate (from the top left of the image): [{},{}] with size = {} pixels.\n\n".format(
                sunspot_Number + 1, sunspot_x_Centre[sunspot_Number],
                sunspot_y_Centre[sunspot_Number],
                sunspot_Size[sunspot_Number])

            print(message)

        sunspot_Number = sunspot_Number + 1
    counter = counter + 1
