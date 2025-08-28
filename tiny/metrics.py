import numpy as np
import logging

logger = logging.getLogger('transformers')

def compute_metrics_text(tokenizer):

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions, labels = predictions[0], labels[0]

        logger.info(f'compute_metrics: raw inputs {predictions.shape=}| {labels.shape=}')

        max_len = max(predictions.shape[-1], labels.shape[-1])

        preds_padded = np.pad(
                        predictions,
                        ((0, 0), (0, max_len - predictions.shape[-1])),
                        constant_values=tokenizer.pad_token_type_id
                    )

        labels_padded = np.pad(
                        labels,
                        ((0, 0), (0, max_len - labels.shape[-1])),
                        constant_values=tokenizer.pad_token_type_id
        )

        token_acc = (preds_padded == labels_padded).mean()

        logger.info(f'compute_metrics: padded {preds_padded.shape=}| {labels_padded.shape=}')

        return {'token_accuracy': token_acc }

    return compute_metrics
