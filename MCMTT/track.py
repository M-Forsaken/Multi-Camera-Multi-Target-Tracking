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
        if other.clusters is not None:
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


class SmoothFeature:
    def __init__(self, learning_rate):
        self.lr = learning_rate
        self.smooth = None

    def __call__(self):
        return self.smooth

    def update(self, embedding):
        if self.smooth is None:
            self.smooth = embedding.copy()
        else:
            self._rolling(self.smooth, embedding, self.lr)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _rolling(smooth, embedding, lr):
        smooth[:] = (1. - lr) * smooth + lr * embedding
        norm_factor = 1. / np.linalg.norm(smooth)
        smooth *= norm_factor

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
        self.is_activated = True
        self.clust_feat = ClusterFeature(num_clusters, metric)

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
                self.clust_feat.update(embedding)
        self.age = 0
        self.hits += 1

    def reinstate(self, frame_id, tlbr, state, embedding):
        self.start_time = frame_id
        self.bboxes.clear()
        self.bboxes.append(tlbr)
        self.estimator_state = state
        if embedding is not None:
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
        if other.clust_feat is not None:
            self.clust_feat.merge(other.clust_feat)
