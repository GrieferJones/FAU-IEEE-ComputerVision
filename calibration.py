import numpy as np
import cv2 as cv
from timeit import default_timer as timer

name = "mono"  # string appended to the matrix and distortion filenames (e.g. "matrixwebcam.txt")

BOARD_DIMS = (6, 8)  # dimensions of the board image we use
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# criteria for subpixel refinement (i think a number of iterations and amount of precision?)
# got this from openCV docs

objpoint = np.zeros((1, BOARD_DIMS[0] * BOARD_DIMS[1], 3), np.float32)
objpoint[0, :, :2] = np.mgrid[0:BOARD_DIMS[0], 0:BOARD_DIMS[1]].T.reshape(-1, 2)
# ^ the 3D "points" in space for the corners of *every* image taken
# looks like [[[0, 0, 0], [1, 0, 0], ... [5, 8, 0]]]
# third coord of each point is assumed to be plane of board, and thus is always zero


def main(args):
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    found = 0   # will be used as a "timestamp" to add a cooldown between detection
    objpoints = []; imgpoints = []  # 3D "points" in space and corresponding "points" on image
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #frame = frame[:, :320 , :]       # ISOLATE LEFT HALF OF INPUT FOR STEREO CAM

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    # convert to grayscale
        ret, corners = cv.findChessboardCorners(gray, BOARD_DIMS, cv.CALIB_CB_ADAPTIVE_THRESH)  # "ret" is a bool flag for if corners are found
        if ret and (timer() - found > 0.25):  # don't append anything until 0.25s cooldown has passed
            # subpixel refinement
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objpoint)  # register match
            imgpoints.append(corners2)
            cv.drawChessboardCorners(frame, BOARD_DIMS, corners2, True) # show points on camera
            print("FOUND #" + str(len(imgpoints)))
            found = timer()     # start short cooldown

        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    # printing results of calibration; rvecs and tvecs discarded because they seem to correspond to each input point
    print("DONE!")
    ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape, None, None)

    print("Camera intrinsic matrix: \n", mtx)
    print("\nDistortion coeffs: \n", dist)
    # print("\nrvecs: \n", rvecs)
    # print("\ntvecs: \n", tvecs)
    np.savetxt("matrix{0}.txt".format(name), mtx)   # save to .txt files after adding set name to filepath
    np.savetxt("dist{0}.txt".format(name), dist)
    # np.savetxt("rvecs.txt", rvecs)
    # np.savetxt("tvecs.txt", tvecs)


if __name__ == '__main__':
    main(0)
