import numpy as np
import cv2 as cv
import math

PURPLE_MIN = np.array([115, 35, 35])        # adjusted for my home use
# PURPLE_MIN = np.array([125, 50, 50])      # adjusted for dice with good lighting
PURPLE_MAX = np.array([160, 255, 255])
CENTER = [slice(230, 250), slice(310, 330)]  # center pixels of my webcam


# generates circular kernels with the specified radius
def circle_kernel(r):
    out = [[1 if math.sqrt(math.pow(i - r, 2) + math.pow(j - r, 2)) <= r else 0 for i in range(2 * r + 1)] for j in
           range(2 * r + 1)]
    return np.array(out, np.uint8)


# kernels used for mask operations (i.e. cleaning stray FPs)
# assumed to be ordered close-open-close; see purple_mask function at bottom
# IF ANYONE IS GOOD AT THIS PLEASE FEEL FREE TO ADD YOUR OWN TO THE DICT BELOW
KERNEL_SETS = {}

# basic square kernels used in docs (with larger numbers for broader opening/closing)
KERNEL_SETS["basic"] = [np.ones((8, 8), np.uint8), np.ones((7, 7), np.uint8), np.ones((6, 6), np.uint8)]
# circular kernels that should theoretically be more natural at larger sizes
KERNEL_SETS["circle"] = [circle_kernel(4), circle_kernel(3), circle_kernel(4)]


def main(args):
    # load dictionary of ArUco/April tags from saved file (see april_setup.py for details)
    fs = cv.FileStorage("markers/aruco_dict.yaml", cv.FILE_STORAGE_READ + cv.FILE_STORAGE_FORMAT_YAML)
    aruco_dict = cv.aruco.Dictionary(0, 6, 2)
    aruco_dict.readDictionary(fs.root())

    params = cv.aruco.DetectorParameters()
    params.useAruco3Detection = True        # this is supposed to be faster than standard detection
    marker_detector = cv.aruco.ArucoDetector(aruco_dict, params)    # detector object is used later

    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    show_angle = False      # whether to show angle display for purple elements
    show_center = False     # whether to show color display for center pixels
    show_marker = 0     # display of April/ArUco markers; '0' shows none, '1' shows detected, '2' shows all candidates
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        mask = purple_mask(frame, show=False, kernel="circle")  # see function
        out = cv.bitwise_and(frame, frame, mask=mask)  # applies mask

        (N, components) = cv.connectedComponents(mask)  # get connected areas in the mask (i.e., get distinct objects)

        # angle display for purple objects
        if show_angle and N > 1:  # if there's only one "region" of mask, then no purple is detected
            index_list = []
            for i in range(1, N):  # go through different region numbers and add into list
                index_list.append(np.argwhere(components == i))

            # center points of each region (mean of coordinates for each); [::-1] just inverts the coords
            points = [np.mean(indices.transpose(), axis=1).astype(np.uint16)[::-1] for indices in index_list]

            for point in points:    # iterate through center points
                x, y, z = get_ray(point, "calibration/matrixwebcam.txt")[0]  # see function

                # vec_angle gets angle between rays; 180/pi converts radians to degrees; copysign assigns correct sign
                angles = (math.copysign(vec_angle([x, z], [0, 1]) * 180 / math.pi, x),
                          math.copysign(vec_angle([y, z], [0, 1]) * 180 / math.pi, y))
                anglestr = "(" + str(round(angles[0], 2)) + ", " + str(round(angles[1], 2)) + ")"  # formats + rounds
                print("POINT:", point)

                cv.putText(frame, anglestr, point, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), lineType=cv.LINE_AA)
                cv.circle(frame, (point[0], point[1]), 30, (255, 255, 255))  # circle center in input and output
                cv.circle(out, (point[0], point[1]), 30, (255, 255, 255))


        # color display for center pixels
        if show_center:
            cv.rectangle(frame, (CENTER[1].start, CENTER[0].start), (CENTER[1].stop, CENTER[0].stop),
                         (255, 255, 255))  # draw rectangle around center pixels
            cv.rectangle(out, (CENTER[1].start, CENTER[0].start), (CENTER[1].stop, CENTER[0].stop),
                         (255, 255, 255))  # draw rectangle around center pixels
            centerhsv = cv.cvtColor(frame[CENTER[0], CENTER[1]], cv.COLOR_BGR2HSV)
            centercolor = np.array([np.round(x, 2) for x in np.mean(np.mean(centerhsv, axis=1),
                                                                    axis=0)])  # get average of these pixels, then round
            cv.putText(frame, str(centercolor), (220, 210), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv.putText(out, str(centercolor), (220, 210), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


        # marker display for april tags
        if show_marker:
            (corners, ids, rejected) = marker_detector.detectMarkers(frame)     # retrieve marker data
            # cv.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(255, 0, 0))  # draw found markers
            if show_marker > 1:
                cv.aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))

            if ids is not None:     # highlight each marker found (if any)
                for (marker, id) in zip(corners, ids):
                    marker = marker[0]  # go down an array layer
                    (xavg, yavg) = (np.mean(marker[:,0]), np.mean(marker[:,1])) # get average coordinates, for display
                    size = min(np.ptp(marker[:,0]), np.ptp(marker[:,1]))        # get rough size of square

                    cv.circle(frame, (int(xavg), int(yavg)), int(size/2.5), (255, 255, 255), cv.FILLED)
                    cv.putText(frame, str(id[0]), (int(xavg - size/5), int(yavg + size/5)), cv.FONT_HERSHEY_SIMPLEX,
                               size/50, (225, 100, 200), 2)


        # Display frames
        cv.imshow('input', frame)
        #cv.imshow('mask', mask)
        cv.imshow('output', out)

        inkey = cv.waitKey(1)
        if inkey == ord('a'):
            show_angle = not show_angle  # toggle angle display
        if inkey == ord('c'):
            show_center = not show_center  # toggle center display
        if inkey == ord('m'):
            show_marker = (show_marker + 1) % 3   # "increment" april/aruco marker display
        if inkey == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def purple_mask(image, show=False, kernel="circle"):
    try:
        kernels = KERNEL_SETS[kernel]     # see kernel sets at top of file
    except KeyError:
        print("WARNING: No kernel set defined with name " + kernel + "; see top of file. Using basic square kernels.")
        kernels = KERNEL_SETS["basic"]

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # convert color scheme
    mask = cv.inRange(hsv, PURPLE_MIN, PURPLE_MAX)  # threshold colors
    mask_close = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernels[0])  # close, open, and close again
    mask_open = cv.morphologyEx(mask_close, cv.MORPH_OPEN, kernels[1])
    mask_final = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernels[2])

    if show:    # show intermediate masks
        cv.imshow('initmask', mask)
        cv.imshow('closedmask', mask_close)
        cv.imshow('openedmask', mask_open)
        cv.imshow('finalmask', mask_final)

    return mask_final


# assorted linear algebra i found online that should convert 2d coords to a 3d vector
# run calibration.py to get the matrix file
def get_ray(point, filename="matrixwebcam.txt"):
    mat1 = np.linalg.inv(np.loadtxt(filename))

    u = point[0]
    v = point[1]
    mat2 = np.array([[u, v, 1]]).transpose()

    ray = np.matmul(mat1, mat2).transpose()
    return ray


# get the angle between two vectors
def vec_angle(v1, v2):
    return math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


if __name__ == '__main__':
    main(0)
