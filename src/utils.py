import math


def iou(boxA, boxB):
    """
    Compute Intersection over Union between two boxes.
    Boxes: (x, y, w, h)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    if boxAArea + boxBArea - interArea == 0:
        return 0

    return interArea / float(boxAArea + boxBArea - interArea)


def center_distance(boxA, boxB):
    """
    Euclidean distance between centers of two boxes.
    """
    ax = boxA[0] + boxA[2] / 2
    ay = boxA[1] + boxA[3] / 2
    bx = boxB[0] + boxB[2] / 2
    by = boxB[1] + boxB[3] / 2

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
