"""
adaptive_ensemble.py

"""

from collections import deque

import numpy as np


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()

        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action


class ChunkedAdaptiveEnsembler:
    """Fuse overlapping action chunks after aligning by execution timestep."""

    def __init__(self, adaptive_ensemble_alpha=0.0):
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_history = []
        self.current_step = 0

    def reset(self):
        self.action_history = []
        self.current_step = 0

    def add_chunk(self, action_chunk):
        action_chunk = np.asarray(action_chunk)
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk with shape (T, D), got {action_chunk.shape}")
        self.action_history.append({"start_step": self.current_step, "actions": action_chunk})

    def step(self):
        self.action_history = [
            item
            for item in self.action_history
            if item["start_step"] + len(item["actions"]) > self.current_step
        ]

        aligned_predictions = []
        for item in self.action_history:
            chunk_index = self.current_step - item["start_step"]
            if 0 <= chunk_index < len(item["actions"]):
                aligned_predictions.append(item["actions"][chunk_index])

        if not aligned_predictions:
            raise RuntimeError(f"No action prediction available for execution step {self.current_step}")

        predictions = np.stack(aligned_predictions)
        if len(predictions) == 1:
            action = predictions[0]
        else:
            # Weight predictions by agreement with the newest chunk. A small
            # alpha stays close to a mean while reducing the influence of a
            # strongly inconsistent old prediction.
            newest = predictions[-1]
            dot_product = np.sum(predictions * newest, axis=1)
            norms = np.linalg.norm(predictions, axis=1) * np.linalg.norm(newest)
            cosine_similarity = dot_product / (norms + 1e-7)
            weights = np.exp(self.adaptive_ensemble_alpha * cosine_similarity)
            weights = weights / weights.sum()
            action = np.sum(weights[:, None] * predictions, axis=0)

        self.current_step += 1
        return action
