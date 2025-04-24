import numpy as np
import cv2 as cv
import cv2.aruco as aruco


def main(args):
    # parameters: bytesList (start empty), marker size (6x6), max # correction bits (4)
    aruco_dict = aruco.Dictionary(0, 6, 4)
    # parameters: number of markers (8), number of bytes per marker (ceil(# pixels / 8)), number of rotations (4)
    aruco_dict.bytesList = np.empty(shape=(9, 5, 4), dtype=np.uint8)

    # list of markers as bits
    bits = np.array([[[1, 1, 0, 1, 0, 1],
                     [0, 1, 1, 1, 0, 1],
                     [0, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [0, 1, 0, 1, 1, 0],
                     [0, 0, 0, 1, 0, 0]],

                    [[1, 1, 0, 1, 1, 0],
                     [0, 1, 0, 1, 1, 1],
                     [1, 1, 1, 1, 0, 0],
                     [0, 1, 1, 0, 0, 0],
                     [1, 0, 1, 1, 0, 1],
                     [0, 0, 1, 0, 0, 1]],

                    [[1, 1, 0, 1, 1, 1],
                     [0, 1, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 1],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0]],

                    [[1, 1, 1, 0, 0, 1],
                     [0, 0, 0, 1, 1, 1],
                     [1, 0, 0, 1, 1, 1],
                     [1, 0, 1, 0, 0, 1],
                     [1, 1, 0, 0, 1, 0],
                     [0, 1, 1, 0, 0, 0]],

                    [[1, 1, 1, 0, 1, 0],
                     [1, 1, 1, 1, 0, 0],
                     [1, 0, 1, 1, 1, 1],
                     [0, 0, 1, 0, 1, 0],
                     [1, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 1, 0]],

                    [[1, 1, 1, 1, 0, 0],
                     [1, 1, 0, 0, 0, 1],
                     [1, 1, 0, 1, 1, 0],
                     [1, 0, 1, 0, 1, 1],
                     [0, 0, 1, 1, 1, 0],
                     [1, 0, 1, 1, 0, 0]],

                    [[0, 0, 0, 0, 0, 1],
                     [0, 1, 0, 1, 1, 0],
                     [1, 0, 1, 0, 0, 1],
                     [0, 1, 1, 1, 0, 1],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 1, 0, 1]],

                    [[0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 0],
                     [0, 1, 0, 1, 0, 0],
                     [1, 0, 1, 1, 1, 0],
                     [0, 0, 0, 1, 1, 1],
                     [0, 1, 0, 1, 0, 0]],

                    [[0, 0, 1, 0, 0, 0],
                     [1, 0, 1, 0, 1, 1],
                     [0, 0, 0, 1, 1, 1],
                     [0, 1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 1, 0],
                     [1, 0, 1, 1, 0, 1]]],
                    dtype=np.uint8)

    for i in range(bits.shape[0]):  # iteratively convert markers to bytelists and add to dictionary
        aruco_dict.bytesList[i] = aruco.Dictionary.getByteListFromBits(bits[i])

    # save markers as images
    for i in range(len(aruco_dict.bytesList)):
        cv.imwrite("markers/m" + str(i) + ".png", aruco.generateImageMarker(aruco_dict, i, 128))

    # save dictionary .yaml file
    aruco_dict.writeDictionary(cv.FileStorage("aruco_dict.yaml", cv.FILE_STORAGE_WRITE + cv.FILE_STORAGE_FORMAT_YAML))


if __name__ == '__main__':
    main(0)