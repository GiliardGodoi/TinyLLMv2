import torch
import logging
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from typing import List
from tiny import CONFIGS

logger = logging.getLogger('transformers')

class DistillationSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self,
                 *args,
                 weights : dict = dict(),
                 default_weight : float = 1.0,
                 **kwargs
                ):
        self.weights = weights
        self.default_weight = default_weight
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask= inputs['attention_mask'],
            labels=inputs['labels']
        )
        sources = inputs['source']
        weights = [self.weights.get(s, self.default_weight) for s in sources ]
        weights = torch.tensor(weights, device=outputs.loss.device)
        loss = outputs.loss @ weights.T
        # ponderar a loss de acordo com source

        return (loss, outputs) if return_outputs else loss