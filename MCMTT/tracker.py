from types import SimpleNamespace
import itertools
import numpy as np

from .track import Track
from .flow import Flow
from .kalman_filter import MeasType, KalmanFilter
from .utils.distance import Metric, iou_dist
from .utils.matching import linear_assignment, greedy_match, fuse_motion, gate_cost
from .utils.rect import as_tlbr, to_tlbr, ios, find_occluded


class MultiTracker:
    def __init__(self, size, hist_tracks, ID_count, metric,
                 max_age=6,
                 age_penalty=2,
                 motion_weight=0.2,
                 assoc_thresh=0.9,
                 reid_thresh=0.45,
                 iou_thresh=0.4,
                 duplicate_thresh=0.8,
                 occlusion_thresh=0.7,
                 conf_thresh=0.5,
                 remove_thresh=0.3,
                 new_thresh=0.5,
                 confirm_hits=1,
                 history_size=50,
                 kalman_filter_cfg=None,
                 flow_cfg=None):
        """Class that uses KLT and Kalman filter to track multiple objects and
        associates detections to tracklets based on motion and appearance.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        metric : {'euclidean', 'cosine'}
            Feature distance metric to associate tracks.
        max_age : int, optional
            Max number of undetected frames allowed before a track is terminated.
            Note that skipped frames are not included.
        age_penalty : int, optional
            Scale factor to penalize KLT measurements for tracks with large age.
        motion_weight : float, optional
            Weight for motion term in matching cost function.
        assoc_thresh : float, optional
            Max matching cost for valid primary association.
        reid_thresh : float, optional
            Max ReID feature dissimilarity for valid reidentification.
        iou_thresh : float, optional
            IoU threshold for association with unconfirmed and unmatched active tracks.
        duplicate_thresh : float, optional
            Track overlap threshold for removing duplicate tracks.
        occlusion_thresh : float, optional
            Detection overlap threshold for nullifying the extracted embeddings for association/reID.
        conf_thresh : float, optional
            Detection confidence threshold for starting a new track.
        confirm_hits : int, optional
            Min number of detections to confirm a track.
        history_size : int, optional
            Max size of track history to keep for reID.
        kalman_filter_cfg : SimpleNamespace, optional
            Kalman Filter configuration.
        flow_cfg : SimpleNamespace, optional
            Flow configuration.
        """
        self.size = size
        self.ID_count = ID_count
        # self.track_high_thresh = args.track_high_thresh
        # self.track_low_thresh = args.track_low_thresh
        # self.new_track_thresh = args.new_track_thresh
        self.metric = Metric[metric.upper()]
        assert max_age >= 1
        self.max_age = max_age
        assert age_penalty >= 1
        self.age_penalty = age_penalty
        assert 0 <= motion_weight <= 1
        self.motion_weight = motion_weight
        assert 0 <= assoc_thresh <= 1
        self.assoc_thresh = assoc_thresh
        assert 0 <= reid_thresh <= 1
        self.reid_thresh = reid_thresh
        assert 0 <= iou_thresh <= 1
        self.iou_thresh = iou_thresh
        assert 0 <= duplicate_thresh <= 1
        self.duplicate_thresh = duplicate_thresh
        assert 0 <= occlusion_thresh <= 1
        self.occlusion_thresh = occlusion_thresh
        assert 0 <= conf_thresh <= 1
        self.conf_thresh = conf_thresh
        assert 0 <= remove_thresh <= 1
        self.remove_thresh = remove_thresh
        assert 0 <= new_thresh <= 1
        self.new_thresh = new_thresh
        assert confirm_hits >= 1
        self.confirm_hits = confirm_hits
        assert history_size >= 0
        self.history_size = history_size

        if kalman_filter_cfg is None:
            kalman_filter_cfg = SimpleNamespace()
        if flow_cfg is None:
            flow_cfg = SimpleNamespace()

        self.tracks = {}
        self.hist_tracks = hist_tracks
        self.kf = KalmanFilter(**vars(kalman_filter_cfg))
        self.flow = Flow(self.size, **vars(flow_cfg))
        self.frame_rect = to_tlbr((0, 0, *self.size))

        self.klt_bboxes = {}

    def reset(self, dt):
        """Reset the tracker for new input context.

        Parameters
        ----------
        dt : float
            Time interval in seconds between each frame.
        """
        self.kf.reset_dt(dt)
        self.hist_tracks.clear()

    def init(self, frame, detections):
        """Initializes the tracker from detections in the first frame.

        Parameters
        ----------
        frame : ndarray
            Initial frame.
        detections : recarray[DET_DTYPE]
            Record array of N detections.
        """
        self.tracks.clear()
        self.flow.init(frame)
        for det in detections:
            if det.conf > self.new_thresh:
                estimator_state = self.kf.create(det.tlbr)
                new_trk = Track(self.ID_count.value, 0, det.tlbr, estimator_state, det.label, self.metric, self.confirm_hits)
                self.ID_count.value += 1
                self.tracks[new_trk.trk_id] = new_trk
                # print(f"{'Detected:':<14}{new_trk}")

    def track(self, frame):
        """Convenience function that combines `compute_flow` and `apply_kalman`.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        self.compute_flow(frame)
        self.apply_kalman()

    def compute_flow(self, frame):
        """Computes optical flow to estimate tracklet positions and camera motion.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        active_tracks = [
            track for track in self.tracks.values() if track.active]
        self.klt_bboxes = self.flow.predict(frame, active_tracks)

    def apply_kalman(self):
        """Performs kalman filter predict and update from KLT measurements.
        The function should be called after `compute_flow`.
        """
        for trk_id, track in list(self.tracks.items()):
            mean, cov = track.estimator_state
            mean, cov = self.kf.predict(mean, cov)
            if trk_id in self.klt_bboxes:
                klt_tlbr = self.klt_bboxes[trk_id]
                # give large KLT uncertainty for occluded tracks
                # usually these with large age and low inlier ratio
                std_multiplier = max(self.age_penalty *
                                     track.age, 1) / track.inlier_ratio
                mean, cov = self.kf.update(
                    mean, cov, klt_tlbr, MeasType.FLOW, std_multiplier)
            next_tlbr = as_tlbr(mean[:4])
            track.update(next_tlbr, (mean, cov))
            if ios(next_tlbr, self.frame_rect) < 0.5:
                if track.confirmed:
                    pass
                    # print(f"{'Out:':<14}{track}")
                self._mark_lost(trk_id)

    def update(self, frame_id, detections, embeddings):
        """Associates detections to tracklets based on motion and feature embeddings.

        Parameters
        ----------
        frame_id : int
            The next frame ID.
        detections : recarray[DET_DTYPE]
            Record array of N detections.
        embeddings : ndarray
            NxM matrix of N extracted embeddings with dimension M.
        """
        occluded_det_mask = find_occluded(
            detections.tlbr, self.occlusion_thresh)
        confirmed_by_depth, unconfirmed = self._group_tracks_by_depth()
        # 1st association with high score detections
        matches1 = []
        u_trk_ids1 = []
        high_score_idx = [idx for idx ,value in enumerate(detections.conf > self.conf_thresh) if value]
        for depth, trk_ids in enumerate(confirmed_by_depth):
            if len(high_score_idx) == 0:
                u_trk_ids1.extend(itertools.chain.from_iterable(
                    confirmed_by_depth[depth:]))
                break
            if len(trk_ids) == 0:
                continue
            detections_high, high_embeddings = detections[high_score_idx], embeddings[high_score_idx]
            cost = self._matching_cost(
                trk_ids, detections_high, high_embeddings)
            matches, u_trk_ids, high_score_idx = linear_assignment(
                cost, trk_ids, high_score_idx)
            matches1 += matches
            u_trk_ids1 += u_trk_ids

        # 2nd association with low score detection
        low_score_idx = np.logical_and(
            (detections.conf > self.remove_thresh), (detections.conf < self.conf_thresh))
        low_score_idx = [idx for idx, value in enumerate(low_score_idx) if value]
        low_score_idx += high_score_idx
        detections_low, low_embeddings = detections[low_score_idx], embeddings[low_score_idx]
        active = [trk_id for trk_id in u_trk_ids1 if self.tracks[trk_id].active]
        u_trk_ids1 = [trk_id for trk_id in u_trk_ids1 if not self.tracks[trk_id].active]
        cost = self.gate(active, detections_low, low_embeddings)
        matches2, u_trk_ids2, u_det_ids= linear_assignment(
            cost, active, low_score_idx)

        # 3rd association with unconfirmed tracks
        u_detections = detections[u_det_ids]
        cost = self._iou_cost(unconfirmed, u_detections)
        matches3, u_trk_ids3, u_det_ids= linear_assignment(
            cost, unconfirmed, u_det_ids)

        # reID with track history
        hist_ids = [trk_id for trk_id, track in self.hist_tracks.items() if track.clust_feat._next_idx > 2]
        u_det_ids = [
            det_id for det_id in u_det_ids if detections[det_id].conf >= self.new_thresh]
        valid_u_det_ids = [
            det_id for det_id in u_det_ids if not occluded_det_mask[det_id]]
        invalid_u_det_ids = [
            det_id for det_id in u_det_ids if occluded_det_mask[det_id]]

        u_detections, u_embeddings = detections[valid_u_det_ids], embeddings[valid_u_det_ids]
        cost = self._reid_cost(hist_ids, u_detections, u_embeddings)

        reid_matches, _, reid_u_det_ids = greedy_match(cost, hist_ids, valid_u_det_ids,
                                                       1 - self.reid_thresh)

        matches = itertools.chain(matches1,matches2,matches3)
        u_trk_ids = itertools.chain(u_trk_ids1,u_trk_ids2, u_trk_ids3)

        # rectify matches that may cause duplicate tracks
        matches, u_trk_ids = self._rectify_matches(
            matches, u_trk_ids, detections)

        # reinstate matched tracks
        for trk_id, det_id in reid_matches:
            if trk_id in self.hist_tracks.keys():
                track = self.hist_tracks.pop(trk_id)
                det = detections[det_id]
                # print(f"{'Reidentified:':<14}{track}")
                estimator_state = self.kf.create(det.tlbr)
                track.reinstate(frame_id, det.tlbr,
                                estimator_state, embeddings[det_id])
                self.tracks[trk_id] = track

        # update matched tracks
        for trk_id, det_id in matches:
            track = self.tracks[trk_id]
            det = detections[det_id]
            mean, cov = self.kf.update(
                *track.estimator_state, det.tlbr, MeasType.DETECTOR)
            next_tlbr = as_tlbr(mean[:4])
            is_valid = not occluded_det_mask[det_id]
            # if track.hits == self.confirm_hits - 1:
            #     print(f"{'Found:':<14}{track}")
            if ios(next_tlbr, self.frame_rect) < 0.5:
                # if track.confirmed:
                #     print(f"{'Out:':<14}{track}")
                self._mark_lost(trk_id)

            track.add_detection(frame_id, next_tlbr,
                                (mean, cov), embeddings[det_id], is_valid)

        # clean up lost tracks
        for trk_id in u_trk_ids:
            track = self.tracks[trk_id]
            track.mark_missed()
            if not track.confirmed:
                # print(f"{'Unconfirmed:':<14}{track}")
                del self.tracks[trk_id]
                continue
            if track.age > self.max_age:
                # print(f"{'Lost:':<14}{track}")
                self._mark_lost(trk_id)
        u_det_ids = itertools.chain(invalid_u_det_ids, reid_u_det_ids)
        # start new tracks
        for det_id in u_det_ids:
            det = detections[det_id]
            if det.conf > self.new_thresh:
                estimator_state = self.kf.create(det.tlbr)
                new_trk = Track(self.ID_count.value, frame_id, det.tlbr, estimator_state,
                                det.label, self.metric, self.confirm_hits)
                self.ID_count.value += 1
                self.tracks[new_trk.trk_id] = new_trk
                # print(f"{'Detected:':<14}{new_trk}")

    def _mark_lost(self, trk_id):
        track = self.tracks.pop(trk_id)
        if track.confirmed:
            self.hist_tracks[trk_id] = track
            if len(self.hist_tracks) > self.history_size:
                self.hist_tracks.popitem()

    def _group_tracks_by_depth(self, group_size=2):
        n_depth = (self.max_age + group_size) // group_size
        confirmed_by_depth = [[] for _ in range(n_depth)]
        unconfirmed = []
        for trk_id, track in self.tracks.items():
            if track.confirmed:
                depth = track.age // group_size
                confirmed_by_depth[depth].append(trk_id)
            else:
                unconfirmed.append(trk_id)
        return confirmed_by_depth, unconfirmed
    
    def _matching_cost(self, trk_ids, detections, embeddings):
        n_trk, n_det = len(trk_ids), len(detections)
        if n_trk == 0 or n_det == 0:
            return np.empty((n_trk, n_det))

        # calculate cost based on clustered features
        cost = np.zeros((n_trk, n_det), dtype=np.float32)
        for i, trk_id in enumerate(trk_ids):
            track = self.tracks[trk_id]
            cost[i:,] = track.clust_feat.embedding_distance(embeddings)

        # fuse motion information
        for row, trk_id in enumerate(trk_ids):
            track = self.tracks[trk_id]
            m_dist = self.kf.motion_distance(
                *track.estimator_state, detections.tlbr)
            fuse_motion(cost[row], m_dist, self.motion_weight)

        # make sure associated pair has the same class label
        t_labels = np.fromiter(
            (self.tracks[trk_id].label for trk_id in trk_ids), int, n_trk)
        gate_cost(cost, t_labels, detections.label,1 - self.assoc_thresh)
        return cost

    def _reid_cost(self, hist_ids, detections, embeddings):
        n_hist, n_det = len(hist_ids), len(detections)
        if n_hist == 0 or n_det == 0:
            return np.empty((n_hist, n_det))

        # calculate cost based on clustered features
        cost = np.zeros((n_hist, n_det), dtype=np.float32)
        for i, trk_id in enumerate(hist_ids):
            track = self.hist_tracks[trk_id] if trk_id in self.hist_tracks.keys() else None
            cost[i:,] = track.clust_feat.embedding_distance(embeddings) if track is not None else 1e5

        t_labels = np.fromiter(
            (t.label for t in self.hist_tracks.values()), int, len(self.hist_tracks.keys()))
        gate_cost(cost, t_labels, detections.label)
        return cost

    def _iou_cost(self, trk_ids, detections):
        n_trk, n_det = len(trk_ids), len(detections)
        if n_trk == 0 or n_det == 0:
            return np.empty((n_trk, n_det))

        t_labels = np.fromiter(
            (self.tracks[trk_id].label for trk_id in trk_ids), int, n_trk)
        t_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids])
        d_bboxes = detections.tlbr
        iou_cost = iou_dist(t_bboxes, d_bboxes)
        gate_cost(iou_cost, t_labels, detections.label, 1 - self.iou_thresh)
        return iou_cost

    def gate(self, trk_ids, detections, embeddings):
        IoU_matrix = self._iou_cost(trk_ids, detections)
        if IoU_matrix.size == 0 or embeddings.size == 0:
            return IoU_matrix
        
        n_trk, n_det = len(trk_ids), len(detections)
        if n_trk == 0 or n_det == 0:
            return np.empty((n_trk, n_det))

        # calculate cost based on clustered features
        cost = np.zeros((n_trk, n_det), dtype=np.float32)
        for i, trk_id in enumerate(trk_ids):
            track = self.tracks[trk_id]
            cost[i:,] = track.clust_feat.embedding_distance(embeddings)

        index = cost > (1 - self.reid_thresh)
        IoU_matrix[index] = 1e5
        return IoU_matrix
    
    def _rectify_matches(self, matches, u_trk_ids, detections):
        matches, u_trk_ids = set(matches), set(u_trk_ids)
        inactive_matches = [
            match for match in matches if not self.tracks[match[0]].active]
        u_active = [trk_id for trk_id in u_trk_ids
                    if self.tracks[trk_id].confirmed and self.tracks[trk_id].active]

        n_inactive_matches = len(inactive_matches)
        if n_inactive_matches == 0 or len(u_active) == 0:
            return matches, u_trk_ids

        m_inactive, det_ids = zip(*inactive_matches)
        t_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in u_active])
        d_bboxes = detections[det_ids,].tlbr
        iou_cost = iou_dist(t_bboxes, d_bboxes)

        col_indices = list(range(n_inactive_matches))
        dup_matches, _, _ = greedy_match(iou_cost, u_active, col_indices,
                                         1. - self.duplicate_thresh)

        for u_trk_id, col in dup_matches:
            m_trk_id, det_id = m_inactive[col], det_ids[col]
            t_u_active, t_m_inactive = self.tracks[u_trk_id], self.tracks[m_trk_id]
            if t_m_inactive.end_time < t_u_active.start_time:
                t_m_inactive.merge_continuation(t_u_active)
                u_trk_ids.remove(u_trk_id)
                del self.tracks[u_trk_id]
            else:
                u_trk_ids.remove(u_trk_id)
                u_trk_ids.add(m_trk_id)
                matches.remove((m_trk_id, det_id))
                matches.add((u_trk_id, det_id))
        return matches, u_trk_ids
