# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
from typing import Callable, List, Tuple

import faiss  # @manual
import numpy as np
from sklearn.preprocessing import normalize
from src.vsc.candidates import CandidateGeneration, MaxScoreAggregation
from src.vsc.index import VideoFeature

logger = logging.getLogger("score_normalization.py")
logger.setLevel(logging.INFO)


def transform_features(
    features: List[VideoFeature], transform: Callable
) -> List[VideoFeature]:
    return [
        dataclasses.replace(feature, feature=transform(feature.feature))
        for feature in features
    ]


def score_normalize(
    queries: List[VideoFeature],
    score_norm_refs: List[VideoFeature],
    l2_normalize: bool = True,
    replace_dim: bool = True,
    beta: float = 1.0,
) -> Tuple[List[VideoFeature], List[VideoFeature]]:
    """
    CSLS style score normalization (as used in the Image Similarity Challenge)
    has the following form. We compute a bias term for each query:

      bias(query) = - beta * sim(query, noise)

    then compute score normalized similarity by incorporating this as an
    additive term for each query:

      sim_sn(query, ref) = sim(query, ref) + bias(query)

    sim(query, ref) is inner product similarity (query * ref), and
    sim(query, noise) is some function of query similarity to a noise dataset
    (score_norm_refs here), such as the similarity to the nearest neighbor.

    We encode the bias term as an extra dimension in the query descriptor,
    and add a constant 1 dimension to reference descriptors, so that inner-
    product similarity is the score-normalized similarity:

      query' = [query bias(query)]
      ref' = [ref 1]
      query' * ref' = (query * ref) + (bias(query) * 1)
          = sim(query, ref) + bias(query) = sim_sn(query, ref)
    """
    if score_norm_refs is not None and replace_dim:
        # Make space for the additional score normalization dimension.
        # We could also use PCA dim reduction, but re-centering can be
        # destructive.
        logger.info("Replacing dimension")
        sn_features = np.concatenate([ref.feature for ref in score_norm_refs], axis=0)
        low_var_dim = sn_features.var(axis=0).argmin()
        queries, score_norm_refs = [
            transform_features(
                x, lambda feature: np.delete(feature, low_var_dim, axis=1)
            )
            for x in [queries, score_norm_refs]
        ]
    if l2_normalize:
        logger.info("L2 normalizing")
        queries, score_norm_refs = [
            transform_features(x, normalize) for x in [queries, score_norm_refs]
        ]
    logger.info("Applying score normalization")
    index = CandidateGeneration(score_norm_refs, MaxScoreAggregation()).index.index
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)

    adapted_queries = []
    # Add the additive normalization term to the queries as an extra dimension.
    for query in queries:
        # KNN search is ok here (versus a threshold/radius/range search) since
        # we're not searching the dataset we're evaluating on.
        similarity, ids = index.search(query.feature, 1)
        norm_term = -beta * similarity[:, :1]
        feature = np.concatenate([query.feature, norm_term], axis=1)
        adapted_queries.append(dataclasses.replace(query, feature=feature))
    return adapted_queries


def score_normalize_with_ref(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    score_norm_refs: List[VideoFeature],
    l2_normalize: bool = True,
    replace_dim: bool = True,
    beta: float = 1.0,
    return_adapted_score_norm_refs: bool = False,
) -> Tuple[List[VideoFeature], List[VideoFeature]]:
    if {f.video_id for f in refs}.intersection({f.video_id for f in score_norm_refs}):
        raise Exception(
            "Normalizing on the dataset we're evaluating on is against VSC rules. "
            "An independent dataset is needed."
        )
    if score_norm_refs is not None and replace_dim:
        # Make space for the additional score normalization dimension.
        # We could also use PCA dim reduction, but re-centering can be
        # destructive.
        logger.info("Replacing dimension")
        sn_features = np.concatenate([ref.feature for ref in score_norm_refs], axis=0)
        low_var_dim = sn_features.var(axis=0).argmin()
        queries, refs, score_norm_refs = [
            transform_features(
                x, lambda feature: np.delete(feature, low_var_dim, axis=1)
            )
            for x in [queries, refs, score_norm_refs]
        ]
    if l2_normalize:
        logger.info("L2 normalizing")
        queries, refs, score_norm_refs = [
            transform_features(x, normalize) for x in [queries, refs, score_norm_refs]
        ]
    logger.info("Applying score normalization")
    index = CandidateGeneration(score_norm_refs, MaxScoreAggregation()).index.index
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)

    adapted_queries = []
    # Add the additive normalization term to the queries as an extra dimension.
    for query in queries:
        # KNN search is ok here (versus a threshold/radius/range search) since
        # we're not searching the dataset we're evaluating on.
        similarity, ids = index.search(query.feature, 1)
        norm_term = -beta * similarity[:, :1]
        feature = np.concatenate([query.feature, norm_term], axis=1)
        adapted_queries.append(dataclasses.replace(query, feature=feature))
    adapted_refs = []
    for ref in refs:
        ones = np.ones_like(ref.feature[:, :1])
        feature = np.concatenate([ref.feature, ones], axis=1)
        adapted_refs.append(dataclasses.replace(ref, feature=feature))
    output = (adapted_queries, adapted_refs)

    if return_adapted_score_norm_refs:
        adapted_score_norm_refs = []
        for score_norm_ref in score_norm_refs:
            ones = np.ones_like(score_norm_ref.feature[:, :1])
            feature = np.concatenate([score_norm_ref.feature, ones], axis=1)
            adapted_score_norm_refs.append(
                dataclasses.replace(score_norm_ref, feature=feature)
            )
        output += (adapted_score_norm_refs,)

    return output


def negative_embedding_subtraction(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    score_norm_refs: List[VideoFeature],
    pre_l2_normalize: bool = False,
    post_l2_normalize: bool = False,
    beta: float = 1.0,
    k: int = 10,
    alpha: float = 1.0,
) -> Tuple[List[VideoFeature], List[VideoFeature]]:
    # impl of https://arxiv.org/abs/2112.04323

    if pre_l2_normalize:
        logger.info("L2 normalizing")
        queries, refs, score_norm_refs = [
            transform_features(x, normalize) for x in [queries, refs, score_norm_refs]
        ]

    logger.info("Applying negative embedding subtraction")
    index = CandidateGeneration(score_norm_refs, MaxScoreAggregation()).index.index
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)

    negative_embeddings = np.concatenate([vf.feature for vf in score_norm_refs], axis=0)

    adapted_queries = []
    for query in queries:
        similarity, ids = index.search(query.feature, k=k)
        weights = similarity[..., None] ** alpha
        topk_negative_embeddings = negative_embeddings[ids] * weights
        subtracted_embedding = topk_negative_embeddings.mean(axis=1) * beta
        adapted_embedding = query.feature - subtracted_embedding
        adapted_queries.append(dataclasses.replace(query, feature=adapted_embedding))

    adapted_refs = []
    for ref in refs:
        similarity, ids = index.search(ref.feature, k=k)
        weights = similarity[..., None] ** alpha
        topk_negative_embeddings = negative_embeddings[ids] * weights
        subtracted_embedding = topk_negative_embeddings.mean(axis=1) * beta
        adapted_embedding = ref.feature - subtracted_embedding
        adapted_refs.append(dataclasses.replace(ref, feature=adapted_embedding))

    if post_l2_normalize:
        logger.info("L2 normalizing")
        adapted_queries, adapted_refs = [
            transform_features(x, normalize) for x in [adapted_queries, adapted_refs]
        ]

    return adapted_queries, adapted_refs
