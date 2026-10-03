import numpy as np


def load_embedding_model(model, device):
    """Load Emotion2Vec only when emotion bridging is requested."""
    from funasr import AutoModel

    return AutoModel(model=model, device=device, hub="hf", disable_update=True)


def utterance_embeddings(model, paths, batch_size):
    """Extract one Emotion2Vec embedding per audio file."""
    for start in range(0, len(paths), batch_size):
        results = model.generate(
            paths[start : start + batch_size],
            granularity="utterance",
            extract_embedding=True,
            batch_size_s=300,
        )
        if len(results) != len(paths[start : start + batch_size]):
            raise ValueError("Emotion2Vec returned an unexpected number of results")
        for result in results:
            embedding = np.asarray(result["feats"], dtype=np.float32)
            yield embedding.mean(axis=0) if embedding.ndim > 1 else embedding


def nearest_donors(query_embeddings, candidate_embeddings):
    candidates = np.array(candidate_embeddings, dtype=np.float32, copy=True)
    queries = np.array(query_embeddings, dtype=np.float32, copy=True)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    similarities = candidates @ queries.T
    indices = similarities.argmax(axis=0)
    scores = similarities[indices, np.arange(len(indices))]
    return indices, scores
