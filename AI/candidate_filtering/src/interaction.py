import math
from config import INTERACTION_DISTANCE


def compute_iou(box1, box2):

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    box1_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    box2_area = max(0, (x4 - x3)) * max(0, (y4 - y3))

    union = box1_area + box2_area - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def normalize_objects(objects):
    normalized = []

    for obj in objects:
        if "track_id" not in obj or "bbox" not in obj:
            continue

        normalized.append({
            "id": obj["track_id"],
            "bbox": obj["bbox"]
        })

    return normalized


def build_interaction_graph(objects):

    objects = normalize_objects(objects)

    nodes = []
    edges = []

    # node 생성
    for obj in objects:
        nodes.append(obj["id"])

    # edge 생성
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):

            box1 = objects[i]["bbox"]
            box2 = objects[j]["bbox"]

            x1, y1, x2, y2 = box1
            x3, y3, x4, y4 = box2

            cx1 = (x1 + x2) / 2.0
            cy1 = (y1 + y2) / 2.0

            cx2 = (x3 + x4) / 2.0
            cy2 = (y3 + y4) / 2.0

            dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

            iou = compute_iou(box1, box2)

            if dist < INTERACTION_DISTANCE or iou > 0.1:
                edges.append((objects[i]["id"], objects[j]["id"]))

    return nodes, edges


def graph_interaction_score(nodes, edges):

    n = len(nodes)

    if n < 2:
        return 0.0

    max_edges = n * (n - 1) / 2.0

    if max_edges == 0:
        return 0.0

    density = len(edges) / max_edges

    return density


def interaction_signal(objects):

    if not objects:
        return 0.0

    nodes, edges = build_interaction_graph(objects)

    score = graph_interaction_score(nodes, edges)

    return score