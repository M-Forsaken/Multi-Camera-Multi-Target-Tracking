from collections import deque
import numpy as np
import numba as nb

from .model_source.label import get_label_name
from .utils.distance import cdist, cosine
from .utils.rect import get_center
from .utils.numba_func import apply_along_axis, normalize_vec


class ClusterFeature:
    def __init__(self, num_clusters, metric):
        self.num_clusters = num_clusters
        self.metric = metric
        self.clusters = None
        self.cluster_sizes = None
        self._next_idx = 0

    def __len__(self):
        return self._next_idx

    def __call__(self):
        return self.clusters[:self._next_idx]

    def update(self, embedding):
        if self._next_idx < self.num_clusters:
            if self.clusters is None:
                self.clusters = np.empty(
                    (self.num_clusters, len(embedding)), embedding.dtype)
                self.cluster_sizes = np.zeros(self.num_clusters, int)
            self.clusters[self._next_idx] = embedding
            self.cluster_sizes[self._next_idx] += 1
            self._next_idx += 1
        else:
            nearest_idx = self._get_nearest_cluster(self.clusters, embedding)
            self.cluster_sizes[nearest_idx] += 1
            self._seq_kmeans(self.clusters, self.cluster_sizes,
                             embedding, nearest_idx)

    def embedding_distance(self, embeddings):
        if self.clusters is None:
            return np.ones(len(embeddings))
        clusters = normalize_vec(self.clusters[:self._next_idx])
        return apply_along_axis(np.min, cdist(clusters, embeddings, self.metric), axis=0)

    def merge(self, other):
        for feature in other.clusters:
            self.update(feature)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_nearest_cluster(clusters, embedding):
        return np.argmin(cosine(np.atleast_2d(embedding), clusters))

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _seq_kmeans(clusters, cluster_sizes, embedding, idx):
        div_size = 1. / cluster_sizes[idx]
        clusters[idx] += (embedding - clusters[idx]) * div_size

class AverageFeature:
    def __init__(self):
        self.sum = None
        self.avg = None
        self.count = 0

    def __call__(self):
        return self.avg

    def is_valid(self):
        return self.count > 0

    def update(self, embedding):
        self.count += 1
        if self.sum is None:
            self.sum = embedding.copy()
            self.avg = embedding.copy()
        else:
            self._average(self.sum, self.avg, embedding, self.count)

    def merge(self, other):
        self.count += other.count
        if self.sum is None:
            self.sum = other.sum
            self.avg = other.avg
        elif other.sum is not None:
            self._average(self.sum, self.avg, other.sum, self.count)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _average(sum, avg, vec, count):
        sum += vec
        div_cnt = 1. / count
        avg[:] = sum * div_cnt
        norm_factor = 1. / np.linalg.norm(avg)
        avg *= norm_factor

class Track:
    def __init__(self, ID, frame_id, tlbr, estimator_state, label, metric, confirm_hits=1, buffer_size=30, num_clusters=4):
        self.trk_id = ID
        self.start_time = frame_id
        self.end_time = frame_id
        self.bboxes = deque([tlbr], maxlen=buffer_size)
        self.confirm_hits = confirm_hits
        self.estimator_state = estimator_state
        self.label = label

        self.age = 0
        self.hits = 0
        # self.avg_feat = AverageFeature()
        self.clust_feat = ClusterFeature(num_clusters,metric)
        self.cross_camera_track = False

        self.last_merge_dis = 1.
        self.inlier_ratio = 1.
        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)

    def __str__(self):
        x, y = get_center(self.tlbr)
        return f'{get_label_name(self.label):<10} {self.trk_id:>3} at ({int(x):>4}, {int(y):>4})'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.end_time - self.start_time

    def __lt__(self, other):
        # ordered by approximate distance to the image plane, closer is greater
        return (self.tlbr[-1], -self.age) < (other.tlbr[-1], -other.age)

    @property
    def tlbr(self):
        return self.bboxes[-1]

    @property
    def active(self):
        return self.age < 2

    @property
    def confirmed(self):
        return self.hits >= self.confirm_hits

    def update(self, tlbr, estimator_state):
        self.bboxes.append(tlbr)
        self.estimator_state = estimator_state

    def add_detection(self, frame_id, tlbr, estimator_state, embedding, is_valid=True):
        self.end_time = frame_id
        self.bboxes.append(tlbr)
        self.estimator_state = estimator_state
        if embedding is not None:
            if is_valid:
                # self.avg_feat.update(embedding)
                self.clust_feat.update(embedding)
        self.age = 0
        self.hits += 1

    def reinstate(self, frame_id, tlbr, state, embedding):
        self.start_time = frame_id
        self.bboxes.clear()
        self.bboxes.append(tlbr)
        self.estimator_state = state
        # self.avg_feat.update(embedding)
        self.clust_feat.update(embedding)
        self.age = 0
        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)

    def mark_missed(self):
        self.age += 1

    def merge_continuation(self, other):
        self.end_time = other.end_time
        self.bboxes.extend(other.bboxes)
        self.estimator_state = other.estimator_state
        self.age = other.age
        self.hits += other.hits

        self.keypoints = other.keypoints
        self.prev_keypoints = other.prev_keypoints

        # self.avg_feat.merge(other.avg_feat)
        self.clust_feat.merge(other.clust_feat)

