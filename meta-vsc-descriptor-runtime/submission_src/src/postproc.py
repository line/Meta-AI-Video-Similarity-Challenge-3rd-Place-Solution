"""
Copyright 2023 LINE Corporation

LINE Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""
from typing import Dict, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.vsc.index import VideoFeature


def _sliding_window_and_concat(
        vfs: List[VideoFeature], stride: int = 1
) -> Dict[str, List[VideoFeature]]:

    new_vfs = []
    kernel = [0.1, 0.2, 0.4, 0.2, 0.1]
    num_stacks = len(kernel)

    for vf in vfs:
        n, d = vf.feature.shape

        num_views = 1
        for i in range(1, len(vf.timestamps)):
            if vf.timestamps[i] <= vf.timestamps[i - 1]:
                num_views += 1

        if n / num_views <= stride:
            continue

        reshaped_feats = vf.feature.reshape(num_views, -1, d)
        reshaped_ts = vf.timestamps.reshape(num_views, -1)

        new_feats = []
        new_timestamps = []
        for i in range(num_views):
            new_feat = reshaped_feats[i]
            new_ts = reshaped_ts[i]
            new_feat = np.concatenate([new_feat[:num_stacks // 2], new_feat, new_feat[-(num_stacks // 2):]], axis=0)
            new_feat = sliding_window_view(new_feat, num_stacks, axis=0)
            assert len(new_feat) == len(reshaped_feats[i])
            if stride > 1:
                new_feat = new_feat[stride // 2::stride]
                new_ts = new_ts[stride // 2::stride]
            weight = np.array(kernel).reshape(1, 1, -1)
            new_feat = (new_feat * weight)
            new_feat = new_feat.transpose(0, 2, 1).reshape(-1, new_feat.shape[1] * num_stacks)
            new_feats.append(new_feat)
            new_timestamps.append(new_ts)

        new_feats = np.concatenate(new_feats, axis=0)
        new_timestamps = np.concatenate(new_timestamps, axis=0)

        new_vfs.append(
            VideoFeature(
                video_id=vf.video_id,
                timestamps=new_timestamps,
                feature=new_feats,
            )
        )

    return new_vfs


def _fit_pca(noises, n_components=512) -> Dict[str, List[VideoFeature]]:
    import faiss
    noise_feats = np.concatenate([vf.feature for vf in noises])
    noise_feats = noise_feats.astype(np.float32)
    mat = faiss.PCAMatrix(noise_feats.shape[-1], n_components)
    mat.train(noise_feats)
    assert mat.is_trained
    return mat


def _apply_pca(vfs, mat) -> Dict[str, List[VideoFeature]]:
    new_vfs = []
    for vf in vfs:
        new_feat = mat.apply(vf.feature.astype(np.float32))
        # new_feat = new_feat / np.linalg.norm(new_feat, axis=-1, keepdims=True)
        new_vfs.append(
            VideoFeature(
                video_id=vf.video_id,
                timestamps=vf.timestamps,
                feature=new_feat,
            )
        )
    return new_vfs


def sliding_pca(
        queries: List[VideoFeature],
        mat: "faiss.PCAMatrix",
        stride: int = 1,
    ) -> List[VideoFeature]:
    queries = _sliding_window_and_concat(queries, stride=stride)
    queries = _apply_pca(queries, mat)
    return queries


def sliding_pca_with_ref(
        queries: List[VideoFeature],
        refs: List[VideoFeature],
        noises: List[VideoFeature],
        stride: int = 1,
        n_components: int = 512,
    ) -> Dict[str, List[VideoFeature]]:

    video_features = {
        'query': _sliding_window_and_concat(queries, stride=stride),
        'ref': _sliding_window_and_concat(refs, stride=stride),
        'noise': _sliding_window_and_concat(noises, stride=stride),
    }
    mat = _fit_pca(video_features['noise'], n_components=n_components)

    for k, vfs in video_features.items():
        video_features[k] = _apply_pca(vfs, mat)

    return video_features, mat
