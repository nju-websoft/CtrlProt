import os

import sys
sys.path.append('./')
import numpy as np
import argparse
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.log_helper import logger_init
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, load_from_disk
import logging
import random
from utils.set_seed import set_seed
import time
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import os
import gc
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import bitsandbytes as bnb
import wandb
from transformers import Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
import json
from trl import DPOTrainer, ORPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import torch.nn as nn

class MLPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop('alpha', args[0] if args else None)

        self.multi_function = kwargs.pop('multi_function', args[0] if args else None)
        
        super().__init__(*args, **kwargs)
        
    
    
    # def zero_out_gradients_hook(self,grad):
    #     """Custom gradient hook to zero out the gradients of the first 200 tokens."""
    #     # print(grad)
    #     grad[:200] = 0
    #     # print(grad)
    #     return grad
    
    
    # def train(self, *args, **kwargs):
    #     # Register the gradient hook here
    #     if self.multi_function:
    #         for name, param in self.model.named_parameters():
    #             if 'prompt_encoder' in name:
    #                 param.register_hook(self.zero_out_gradients_hook)
        
    #     # Call the parent class's on_train_begin if necessary
    #     return super().train(*args, **kwargs)
    
    
    def tokenize_row(self, *args, **kwargs):
        # print("Custom train method")
        feature = kwargs.get('feature', args[0] if args else None)
            
        batch = super().tokenize_row(*args, **kwargs)
        if feature:
            batch['chosen_score'] = feature['chosen_score']
            batch["chosen_weight"] = feature["chosen_weight"]
            batch['rejected_score'] = feature['rejected_score']
            batch["rejected_weight"] = feature["rejected_weight"] 
            batch["regularization_term"] = feature["regularization_term"] 
            
        return batch
    
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)
                    
                    
        regularization_term = torch.tensor(batch["regularization_term"]).to(policy_chosen_logps.device)
        losses, chosen_rewards, rejected_rewards = self.mlpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            regularization_term
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        # lambda_weight = torch.tensor(batch["lambda_weight"]).to(losses.device)
        # losses = losses * lambda_weight
        
        return losses.mean(), metrics
    
    def mlpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        regularization_term: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits - self.alpha * regularization_term) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits + self.alpha * regularization_term) * self.label_smoothing
            )


            
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        losses2 = (
                -F.logsigmoid(self.beta * (logits )) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * (logits )) * self.label_smoothing
            )


        
        return losses, chosen_rewards, rejected_rewards
        