import numpy as np

def compute_metrics_text(tokenizer):
    """
    Defines a function for computing custom evaluation metrics.

    Args:
        tokenizer: The tokenizer used for decoding model predictions.

    Returns:
        A function that computes metrics based on predictions and labels.
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions[0] = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
        tokenizer.batch_decode(predictions[0], skip_special_tokens=True)
        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Compute accuracy
        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))
        return {'accuracy': acc}

    return compute_metrics
