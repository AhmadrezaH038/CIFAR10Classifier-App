def select_model(results):
    """
    results = list of (model_name, pred_idx, confidence)
    returns the tuple with highest confidence
    """
    if not results:
        return None
    return max(results, key=lambda x: x[2])