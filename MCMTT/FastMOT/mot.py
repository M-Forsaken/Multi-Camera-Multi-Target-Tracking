from types import SimpleNamespace
from enum import Enum
import logging
import numpy as np
import numba as nb
import cv2

from object_detection import Detection
from feature_extractor import FeatureExtractor
from tracker import MultiTracker
from utils.visualization import Visualizer
from utils.numba_func import bisect_right


LOGGER = logging.getLogger(__name__)

class MOT:
    def __init__(self, size,
                 detector_frame_skip=1,
                 class_ids=(0,),
                 yolo_detector_cfg=None,
                 feature_extractor_cfgs=None,
                 tracker_cfg=None,
                 visualizer_cfg=None,
                 draw=True):
        """Top level module that integrates detection, feature extraction,
        and tracking together.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        detector_type : {'SSD', 'YOLO', 'public'}, optional
            Type of detector to use.
        detector_frame_skip : int, optional
            Number of frames to skip for the detector.
        class_ids : sequence, optional
            Class IDs to track. Note class ID starts at zero.
        ssd_detector_cfg : SimpleNamespace, optional
            SSD detector configuration.
        yolo_detector_cfg : SimpleNamespace, optional
            YOLO detector configuration.
        public_detector_cfg : SimpleNamespace, optional
            Public detector configuration.
        feature_extractor_cfgs : List[SimpleNamespace], optional
            Feature extractor configurations for all classes.
            Each configuration corresponds to the class at the same index in sorted `class_ids`.
        tracker_cfg : SimpleNamespace, optional
            Tracker configuration.
        visualizer_cfg : SimpleNamespace, optional
            Visualization configuration.
        draw : bool, optional
            Draw visualizations.
        """
        self.size = size
        self.detector_type = 1
        assert detector_frame_skip >= 1
        self.detector_frame_skip = detector_frame_skip
        self.class_ids = tuple(np.unique(class_ids))
        self.draw = draw
        if yolo_detector_cfg is None:
            yolo_detector_cfg = SimpleNamespace()
        if feature_extractor_cfgs is None:
            feature_extractor_cfgs = (SimpleNamespace(),)
        if tracker_cfg is None:
            tracker_cfg = SimpleNamespace()
        if visualizer_cfg is None:
            visualizer_cfg = SimpleNamespace()
        if len(feature_extractor_cfgs) != len(class_ids):
            raise ValueError('Number of feature extractors must match length of class IDs')

        print('Loading feature extractor models...')
        self.extractors = [FeatureExtractor(**vars(cfg)) for cfg in feature_extractor_cfgs]
        self.tracker = MultiTracker(self.size, self.extractors[0].metric, **vars(tracker_cfg))
        self.visualizer = Visualizer(**vars(visualizer_cfg))
        self.frame_count = 0

    def visible_tracks(self):
        """Retrieve visible tracks from the tracker

        Returns
        -------
        Iterator[Track]
            Confirmed and active tracks from the tracker.
        """
        return (track for track in self.tracker.tracks.values()
                if track.confirmed and track.active)

    def reset(self, cap_dt):
        """Resets multiple object tracker. Must be called before `step`.

        Parameters
        ----------
        cap_dt : float
            Time interval in seconds between each frame.
        """
        self.frame_count = 0
        self.tracker.reset(cap_dt)

    def step(self, frame):
        """Runs multiple object tracker on the next frame.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        detections = []
        cls_bboxes = []
        if self.frame_count == 0:
            detections = Detection(frame)
            self.tracker.init(frame, detections)
        elif self.frame_count % self.detector_frame_skip == 0:
            self.tracker.compute_flow(frame)
            detections = Detection(frame)
            cls_bboxes = self._split_bboxes_by_cls(detections.tlbr, detections.label,
                                                        self.class_ids)
            for extractor, bboxes in zip(self.extractors, cls_bboxes):
                extractor.extract_async(frame, bboxes)
            self.tracker.apply_kalman()
            embeddings = []
            for extractor in self.extractors:
                embeddings.append(extractor.postprocess())
            embeddings = np.concatenate(embeddings) if len(embeddings) > 1 else embeddings[0]
            self.tracker.update(self.frame_count, detections, embeddings)
        else:
            self.tracker.track(frame)
        if self.draw:
            self._draw(frame, detections)
        self.frame_count += 1

    @staticmethod
    @nb.njit(cache=True)
    def _split_bboxes_by_cls(bboxes, labels, class_ids):
        cls_bboxes = []
        begin = 0
        for cls_id in class_ids:
            end = bisect_right(labels, cls_id, begin)
            cls_bboxes.append(bboxes[begin:end])
            begin = end
        return cls_bboxes

    def _draw(self, frame, detections):
        visible_tracks = list(self.visible_tracks())
        self.visualizer.render(frame, visible_tracks, detections, self.tracker.klt_bboxes.values(),
                               self.tracker.flow.prev_bg_keypoints, self.tracker.flow.bg_keypoints)
        cv2.putText(frame, f'visible: {len(visible_tracks)}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
