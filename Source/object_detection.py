from config import *
import cv2
from SORT_Algoritm import *
from threading import Thread

from operator import itemgetter

tracker = SortTracker(max_age=sort_max_age, min_hits=sort_min_hits,
                      iou_threshold=sort_iou_thresh)
rand_color_list = random_color_list()


def draw_boxes(frame, object_list,camera_id = None, names = None):
    for object in object_list:
        x1, y1, x2, y2 = [int(item) for item in object[:4]]
        id = int(object[4])
        cat = int(object[5]) 
        label = names[cat]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 20), 1)
        cv2.putText(frame, "id" + str(id), (x1 - 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 2)
    if trail:
        tracks =tracker.getTrackers()
        for track in tracks:
            if track.camera_id == camera_id: 
                [cv2.line(frame, (int(track.centroidarr[i][0]),
                            int(track.centroidarr[i][1])), 
                            (int(track.centroidarr[i+1][0]),
                            int(track.centroidarr[i+1][1])),
                            rand_color_list[track.id], thickness=3) 
                            for i,_ in  enumerate(track.centroidarr) 
                                if i < len(track.centroidarr)-1 ] 

    return frame

def Object_tracking(object_list):
    """
        Return a numpy array of objects in [x1, y1, x2, y2, id, class_id,camera_id] format 
    """
    if len(object_list) == 0:
        return []
    list = []
    dets_to_sort = np.empty((0, 6))
    for item in object_list:
        dets_to_sort = np.vstack((dets_to_sort,
                                  item))
    tracked_dets = tracker.update(dets_to_sort)
    for track in tracked_dets:
        list.append(np.concatenate((track[:4], [track[9]], [track[4]],[track[5]])))
    return np.array(list)
def Detection(count,frame):
    results=model(frame, verbose=False)
    for item in results:
        object_list = []
        for data in item.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = data
            if model.names[class_id] == tag and score >= confidence:
                object_list.append(
                        np.array([x1, y1, x2, y2, class_id,count]))
    return object_list

def ObjectDetection(Frame_list):
    Frame_bbox = []
    bbox_list = []
    if len(Frame_list) != 0:
        for count,frame in enumerate(Frame_list):
            bbox_list.extend(Detection(count,frame))
        if len(bbox_list) != 0:
            bbox_list = np.array(sorted(Object_tracking(bbox_list), key=itemgetter(6)))
            temp = 0
            temp_list = []
            for bbox in bbox_list:
                if bbox[6] != temp:
                    temp = bbox[6]
                    if len(temp_list) != 0:
                        Frame_bbox.append(np.array(temp_list))
                    temp_list = []
                temp_list.append(bbox)
            Frame_bbox.append(np.array(temp_list))
        for count,frame in enumerate(Frame_list):
            if count <= len(Frame_bbox) - 1:
                draw_boxes(frame, Frame_bbox[count],count, model.names)
            cv2.putText(frame, f"Cam {count + 1} FPS: {int(clock.get_fps())}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("CCTV "+ str(count), frame)
            cv2.waitKey(1)

