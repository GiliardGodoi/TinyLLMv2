import torch
import logging
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from typing import List
from tiny import CONFIGS

logger = logging.getLogger('transformers')
teachers = CONFIGS['teachers']

class MultiTeacherDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    def __init__(self, *args, student_key='student', teachers_keys=teachers, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_key = student_key
        self.teacher_keys = teachers_keys
        self.all_data_keys = [student_key] + teachers_keys

    def __call__(self, features, return_tensors=None):

        batch_roles = { role : [] for role in self.all_data_keys }
        for feat in features:
            for role in self.all_data_keys:
                batch_roles[role].append({
                    'input_ids' : feat[f'{role}_input_ids'],
                    'attention_mask' : feat[f'{role}_attention_mask'],
                    'labels' : feat[f'{role}_labels']
                })

        output = {
            role : super().__call__(batch_roles[role], return_tensors) for role in self.all_data_keys
        }

        return output

class MultiTeacherSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self,
                 student_weight : float,
                 teachers_weights : List[float],
                 output_rationale,
                 *args,
                 student_key='student',
                 teachers_keys=teachers,
                 **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.student_weight = student_weight
        self.teachers_weights = teachers_weights
        self.output_rationale = output_rationale
        self.student_key = student_key
        self.teacher_keys = teachers_keys

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        all_outputs = dict()
        student_key = self.student_key
        teacher_keys = self.teacher_keys

        student_inputs = inputs[student_key]
        student_outputs = model(**student_inputs)

        # logger.info(f"{student_outputs.loss} | {student_inputs['input_ids'].shape} | {student_inputs['attention_mask'].shape} | {student_inputs['labels'].shape}")
        logger.info(f"{student_outputs.loss.item()=}")

        all_outputs[student_key] = student_outputs
        losses_values = [ student_outputs.loss * self.student_weight ]

        for teacher, weight in zip(teacher_keys, self.teachers_weights):
            teacher_inputs = inputs[teacher]
            teacher_outputs = model(**teacher_inputs)
            losses_values.append(teacher_outputs.loss * weight)
            all_outputs[teacher] = teacher_outputs

        losses = torch.stack(losses_values)
        final_loss = torch.sum(losses.to(torch.float32))

        logger.info(f"{final_loss.item()=} | {len(all_outputs)}")

        return (final_loss, all_outputs) if return_outputs else final_loss

    def prediction_step(
            self, model, inputs, prediction_loss_only, ignore_keys=None
        ):

        prediction_outputs = dict()

        student_input = inputs[self.student_key]
        student_pred = super().prediction_step(
                                model, student_input, prediction_loss_only=False, ignore_keys=ignore_keys
                            )
        loss, logits, labels = student_pred
        loss *= self.student_weight
        if not torch.isfinite(loss):
            logger.warning(f"Student loss is NaN/inf: {loss.item()}")
            loss = torch.zeros_like(loss)
        prediction_outputs[self.student_key] = (loss, logits, labels)

        if self.output_rationale:
            for teacher, weight in zip(self.teacher_keys, self.teachers_weights):
                teacher_input = inputs[teacher]
                teacher_pred = super().prediction_step(
                    model, teacher_input, prediction_loss_only=False, ignore_keys=ignore_keys
                )
                loss, logits, labels = teacher_pred
                loss *= weight
                if not torch.isfinite(loss):
                    logger.warning(f"Teacher '{teacher}' loss is NaN/inf: {loss.item()}")
                    loss = torch.zeros_like(loss)
                prediction_outputs[teacher] = (loss, logits, labels)

        losses = torch.stack([pred[0] for pred in prediction_outputs.values()])
        final_loss = torch.sum(losses)

        if not torch.isfinite(final_loss):
            logger.warning(f"Final loss is NaN/inf: {final_loss.item()}")
            final_loss = torch.zeros_like(final_loss)

        final_logits = [pred[1] for pred in prediction_outputs.values()]
        final_labels = [pred[2] for pred in prediction_outputs.values()]

        logger.info(f"{final_loss=} | {len(final_logits)} | {len(final_labels)}")

        return final_loss, final_logits, final_labels