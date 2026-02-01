import math

def center_of_bbox(bbox):
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area

    return inter_area / union_area if union_area > 0 else 0

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
