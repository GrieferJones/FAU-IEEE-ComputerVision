import numpy as np
import pandas as pd
import cv2 as cv
import math
import random

WIDTH = 320     # width of EACH webcam
HEIGHT = 240

PURPLE_MIN = np.array([110, 65, 35])        # adjusted for my home use
# PURPLE_MIN = np.array([125, 50, 50])      # adjusted for dice with good lighting
PURPLE_MAX = np.array([150, 255, 255])
#CENTER = [slice(HEIGHT/2 - 10, HEIGHT/2 + 10), slice(WIDTH/2 - 10, WIDTH/2 + 10)]  # center pixels of each cam
CENTER = [slice(110, 130), slice(150, 170)]  # center pixels of stereo cam

MIN_AREA = 50           # minimum pixel area of purple region to be visible
CYCLE_COOLDOWN = 100    # number of frames between output updates, if "slow" is on


# generates circular kernels with the specified radius
def circle_kernel(r):
    out = [[1 if math.sqrt(math.pow(i - r, 2) + math.pow(j - r, 2)) <= r else 0 for i in range(2 * r + 1)] for j in
           range(2 * r + 1)]
    return np.array(out, np.uint8)


# kernels used for mask operations (i.e. cleaning stray FPs)
# assumed to be ordered close-open-close; see purple_mask function at bottom
KERNEL_SETS = {}

# basic square kernels used in docs (with larger numbers for broader opening/closing)
KERNEL_SETS["basic"] = [np.ones((8, 8), np.uint8), np.ones((7, 7), np.uint8), np.ones((6, 6), np.uint8)]
# circular kernels that should theoretically be more natural at larger sizes
KERNEL_SETS["circle"] = [circle_kernel(4), circle_kernel(3), circle_kernel(4)]


def main(args):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cooldown = 0
    slow = True  # whether info and frames are updated on 25-frame cooldown or on every frame
    print_stats = False

    while True:
        # cv.waitKey(-1)         # uncomment to go frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frames = [frame[:,:WIDTH,:], frame[:,WIDTH:,:]]
        masks = [purple_mask(frames[0], show=False, kernel="circle"), purple_mask(frames[1], show=False, kernel="circle")]  # see function
        outs = [cv.bitwise_and(frames[0], frames[0], mask=masks[0]), cv.bitwise_and(frames[1], frames[1], mask=masks[1])]  # applies mask

        components = (cv.connectedComponentsWithStats(masks[0]), cv.connectedComponentsWithStats(masks[1]))  # get connected areas in the mask (i.e., get distinct objects)

        (NL, NR) = (components[0][0] - 1, components[1][0] - 1)     # number of purple regions in each camera
        dataL = []; dataR = []      # array of regions in each camera

        # regions in each camera
        for i in range(1, NL + 1):
            pixels = np.argwhere(components[0][1] == i)
            # commented out for poor performance
            #color = np.mean([frames[0][j][k] for j in range(HEIGHT) for k in range(WIDTH)], axis=0)
            dataL.append(Region(pixels, components[0][2][i], components[0][3][i]))

        for i in range(1, NR + 1):
            pixels = np.argwhere(components[1][1] == i)
            #color = np.mean([frames[1][j][k] for j in range(HEIGHT) for k in range(WIDTH)], axis=0)
            dataR.append(Region(pixels, components[1][2][i], components[1][3][i]))

        dataL = [r for r in dataL if r.area > MIN_AREA]         # filter out very small areas
        dataR = [r for r in dataR if r.area > MIN_AREA]

        (dataL, dataR) = match(dataL, dataR)
        data = (dataL, dataR)

        for i in range(max(len(data[0]), len(data[1]))):      # for each potential region in halves
            for half in (0,1):      # print individually for left and right halves
                if i >= len(data[half]):
                    continue

                region = data[half][i]
                point = [int(c) for c in region.centroid]
                (left, top, width, height) = region.position
                cv.rectangle(frames[half], (left, top), (left+width, top+height), random_color(i))      # mark bounding box with set-seed random color
                cv.rectangle(outs[half], (left, top), (left+width, top+height), random_color(i))

                angles = get_cam_angles(region.centroid)

                if print_stats >= 1:
                    cv.circle(frames[half], (point[0], point[1]), 3, random_color(i), cv.FILLED)
                    cv.circle(outs[half], (point[0], point[1]), 3, random_color(i), cv.FILLED)

                if print_stats == 1:
                    anglestr = "(" + str(round(angles[0], 2)) + ", " + str(round(angles[1], 2)) + ")"  # formats/rounds

                    cv.putText(frames[half], anglestr, [region.left, region.top], cv.FONT_HERSHEY_SIMPLEX,
                               math.sqrt(region.area) / 50, random_color(i), lineType=cv.LINE_AA)
                    cv.putText(outs[half], anglestr, [region.left, region.top], cv.FONT_HERSHEY_SIMPLEX,
                               math.sqrt(region.area) / 50, random_color(i), lineType=cv.LINE_AA)


            if i < len(data[0]) and i < len(data[1]):   # if region is in both halves
                distance = round(get_object_distance(data[0][i], data[1][i], 0.5), 2)
                if print_stats == 2:
                    cv.circle(frames[half], (point[0], point[1]), 3, random_color(i), cv.FILLED)
                    cv.circle(outs[half], (point[0], point[1]), 3, random_color(i), cv.FILLED)

                    for half in (0,1):
                        cv.putText(frames[half], str(distance), [data[half][i].left, data[half][i].top], cv.FONT_HERSHEY_SIMPLEX,
                                   math.sqrt(dataL[i].area) / 30, random_color(i), lineType=cv.LINE_AA)
                        cv.putText(outs[half], str(distance), [data[half][i].left, data[half][i].top], cv.FONT_HERSHEY_SIMPLEX,
                                   math.sqrt(dataR[i].area) / 30, random_color(i), lineType=cv.LINE_AA)


        if (slow and cooldown <= 0) or not slow:    # print/display statements controlled by "speed"
            cooldown = CYCLE_COOLDOWN

            # Display frames
            cv.imshow("LEFT", frames[0])
            cv.imshow("RIGHT", frames[1])
            cv.imshow("MASKLEFT", outs[0])
            cv.imshow("MASKRIGHT", outs[1])

        cooldown -= 1

        inkey = cv.waitKey(1)
        if inkey == ord('q'):
            break
        if inkey == ord('s'):
            slow = not slow
        if inkey == ord('p'):
            print_stats = (print_stats + 1) % 3

    cap.release()
    cv.destroyAllWindows()


# # # # # ANGLE MATH FUNCTIONS

# assorted linear algebra i found online that should convert 2d coords to a 3d vector
# run calibration.py to get the matrix file
def get_ray(point, filename="matrixstereo.txt"):
    mat1 = np.linalg.inv(np.loadtxt(filename))

    u = point[0]
    v = point[1]
    mat2 = np.array([[u, v, 1]]).transpose()

    ray = np.matmul(mat1, mat2).transpose()
    return ray


# get the angle between two vectors
def vec_angle(v1, v2):
    return math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# get the x and y angle between the point and the camera, in degrees
def get_cam_angles(point, filename="matrixstereo.txt", radians = False):
    x, y, z = get_ray(point, filename)[0]  # see function

    # vec_angle gets angle between rays; 180/pi converts radians to degrees; copysign assigns correct sign
    return (math.copysign(vec_angle([x, z], [0, 1]) * 180 / math.pi, x),
            math.copysign(vec_angle([y, z], [0, 1]) * 180 / math.pi, y))


# given two regions, calculates the average distance between object and each camera in cm
def get_object_distance(regionL, regionR, cam_distance = 6.05):
    centroid1 = regionL.centroid; centroid2 = regionR.centroid
    (xL, yL) = get_cam_angles(centroid1); (xR, yR) = get_cam_angles(centroid2)      # x angles read from camera
    xLt = 90 - xL
    xRt = 90 + xR

    C = 180 - xLt - xRt     # angle between both cameras, from object

    dL = cam_distance * math.sin(xLt * math.pi / 180) / math.sin(C * math.pi / 180)     # *x* distances from each camera to object
    dR = cam_distance * math.sin(xRt * math.pi / 180) / math.sin(C * math.pi / 180)

    dL /= math.cos(yL * math.pi / 180)      # adjust distances to incorporate y angles
    dR /= math.cos(yR * math.pi / 180)

    return (dL + dR) / 2

# # # # # REGION MATCHING FUNCTIONS

# given two region instances, calculates the "difference" between them from manhattan distance and area difference
# x-axis difference is not considered by default since cameras are aligned on this axis
def get_difference(r1, r2, x=0, y=1, a=1, c=1):
    (x1, y1) = r1.centroid;
    (x2, y2) = r2.centroid

    x_score = abs(x1 - x2 - 50) * 100 * x  # pos. in right should be less than in left
    y_score = abs(y1 - y2) * 100 * y
    area_score = abs(r1.area - r2.area) * a

    #color_score = (abs(r1.color[0] - r2.color[0]) + abs(r1.color[1] - r2.color[1]) + abs(r1.color[2] - r2.color[2])) * 10 * c
    #print(x_score, y_score, area_score, color_score)       # difference of average colors

    if (r1.left == 0 and r2.left == 0) or (r1.left + r1.width == WIDTH and r2.left + r2.width == WIDTH):
        x_score = 0; area_score /= 2    # cut some slack if both objects on left/right

    return x_score + y_score + area_score   # + color_score  # , x_score, y_score, area_score, color_score


# given arrays of regions (with possibly different sizes), reorders arrays to match up indices of corresponding regions
def match(left, right, x_factor = 0):
    if len(left) == 0 or len(right) == 0:
        return left, right  # if one array is empty, no reordering needed

    minimum = (999999, 0, 0)     # keeps track of minimum difference, and indices of it
    for l in range(len(left)):
        for r in range(len(right)):
            diff = get_difference(left[l], right[r], x_factor)
            if diff < minimum[0]:
                minimum = (diff, l, r)

    (_, l, r) = minimum
    newleft = left[:l] + left[l+1:]         # remove selected indices from arrays
    newright = right[:r] + right[r+1:]
    newleft, newright = match(newleft, newright)    # recurse over smaller arrays
    return [left[l]] + newleft, [right[r]] + newright  # insert ordered values around recursed result


# # # # # MISC FUNCTIONS

class Region:
    region = []  # list of coordinates within the region
    centroid = []  # coordinates of region centroid
    area = 0  # number of pixels within region
    position = ()  # left edge, top edge, width, and height in pixels
    color = ()  # average hsv value of pixels
    x_index = -1

    def __init__(self, r, stats, centroid, color=(), index=-1):
        self.region = r
        (self.left, self.top, self.width, self.height, area) = stats
        self.area = area
        self.position = (self.left, self.top, self.width, self.height)
        self.centroid = centroid
        self.color = color
        self.x_index = index

    def set_index(self, index):
        self.x_index = index

    def set_color(self, color):
        self.color = color


def purple_mask(image, show=False, kernel="circle"):
    try:
        kernels = KERNEL_SETS[kernel]  # see kernel sets at top of file
    except KeyError:
        print("WARNING: No kernel set defined with name " + kernel + "; see top of file. Using basic square kernels.")
        kernels = KERNEL_SETS["basic"]

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # convert color scheme
    mask = cv.inRange(hsv, PURPLE_MIN, PURPLE_MAX)  # threshold colors
    mask_close = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernels[0])  # close, open, and close again
    mask_open = cv.morphologyEx(mask_close, cv.MORPH_OPEN, kernels[1])
    mask_final = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernels[2])

    if show:  # show intermediate masks
        cv.imshow('initmask', mask)
        cv.imshow('closedmask', mask_close)
        cv.imshow('openedmask', mask_open)
        cv.imshow('finalmask', mask_final)

    return mask_final


# returns tuple of three uniformly random values; hue between 0 and 255, sat and val between 50 and 255
def random_color(seed):
    random.seed(seed)
    return random.randrange(0, 255), random.randrange(50, 255), random.randrange(50, 255)


main(0)
