# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import torch
from typing import List

import numpy as np

from src.vsc.index import VideoFeature
from src.vsc.metrics import CandidatePair, Match


class Localization(abc.ABC):
    @abc.abstractmethod
    def localize(self, candidate: CandidatePair) -> List[Match]:
        pass

    def localize_all(self, candidates: List[CandidatePair]) -> List[Match]:
        matches = []
        for candidate in candidates:
            matches.extend(self.localize(candidate))
        return matches


class LocalizationWithMetadata(Localization):
    def __init__(self, queries: List[VideoFeature], refs: List[VideoFeature]):
        self.queries = {m.video_id: m for m in queries}
        self.refs = {m.video_id: m for m in refs}

    def similarity(self, candidate: CandidatePair):
        a = self.queries[candidate.query_id].feature
        b = self.refs[candidate.ref_id].feature
        return np.matmul(a, b.T)
    
    def count_views(self, candidate: CandidatePair):
        a = self.queries[candidate.query_id].timestamps
        q_views = np.count_nonzero(a == a[0])
        b = self.refs[candidate.ref_id].timestamps
        r_views = np.count_nonzero(b == b[0])
        return [q_views, r_views]


class VCSLLocalization(LocalizationWithMetadata):
    def __init__(self, queries, refs, model_type, similarity_bias=0.0, **kwargs):
        super().__init__(queries, refs)

        # Late import: allow OSS use without VCSL installed
        from src.vsc.vta import build_vta_model  # @manual

        self.model = build_vta_model(model_type, **kwargs)
        self.similarity_bias = similarity_bias

    def similarity(self, candidate: CandidatePair):
        """Add an optional similarity bias.

        Some localization methods do not tolerate negative values well.
        """
        return super().similarity(candidate) + self.similarity_bias

    def localize_all(self, candidates: List[CandidatePair]) -> List[Match]:
        # sims = [(f"{c.query_id}-{c.ref_id}", self.similarity(c)) for c in candidates]
        sims = [(f"{c.query_id}-{c.ref_id}", self.count_views(c), self.similarity(c)) for c in candidates] 
        results = self.model.forward_sim(sims)
        assert len(results) == len(candidates)
        matches = []
        for (candidate, (key, _, sim), result) in zip(candidates, sims, results):
            query: VideoFeature = self.queries[candidate.query_id]
            ref: VideoFeature = self.refs[candidate.ref_id]
            assert key == result[0]
            for box in result[1]:
                (x1, y1, x2, y2, score) = box
                match = Match(
                    query_id=candidate.query_id,
                    ref_id=candidate.ref_id,
                    query_start=query.get_timestamps(x1)[0],
                    query_end=query.get_timestamps(x2)[1],
                    ref_start=ref.get_timestamps(y1)[0],
                    ref_end=ref.get_timestamps(y2)[1],
                    score=0.0,
                )
                # score = self.score(candidate, match, box, sim)
                match = match._replace(score=score)
                matches.append(match)
        return matches

    def localize(self, candidate: CandidatePair) -> List[Match]:
        return self.localize_all([candidate])

    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        return 1.0


class VCSLLocalizationMaxSim(VCSLLocalization):
    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        x1, y1, x2, y2, _score = box
        return similarity[x1:x2, y1:y2].max() - self.similarity_bias

class VCSLLocalizationMatchScore(VCSLLocalization):
    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        x1, y1, x2, y2, _score = box
        return _score


class VCSLLocalizationCandidateScore(VCSLLocalization):
    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        return candidate.score
