from config import *
import cv2

rand_color_list = random_color_list()

DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


def draw_boxes(frame, object_list, camera_id=None):
    for object in object_list:
        x1, y1, x2, y2 = object.tlbr.astype(np.int_)
        id = object.conf
        label = int(object.label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 20), 1)
        cv2.putText(frame, "id" + str(names[label]), (x1 - 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 2)
    return frame
def Detection(frame):
    results = model(frame ,conf = confidence,iou = 0.5,verbose=False)
    for item in results:
        object_list = []
        for data in item.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = data
            if names[class_id] == tag:
                object_list.append(
                    np.array(((x1, y1, x2, y2), (class_id), (score)), dtype=DET_DTYPE))
    return np.array(object_list, dtype=DET_DTYPE).view(np.recarray)
def ObjectDetection(frame):
    bbox_list = []
    bbox_list = Detection(frame)
    draw_boxes(frame, bbox_list)
    cv2.putText(frame, f"FPS: {int(clock.get_fps())}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow("CCTV", frame)
    cv2.waitKey(1)
