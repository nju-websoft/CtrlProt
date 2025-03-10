import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
import json
import logging
import os
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from accelerate import PartialState
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from safetensors.torch import load_file as safe_load_file
from transformers import PreTrainedModel
from trl import AutoModelForSeq2SeqLMWithValueHead
from trl.import_utils import is_npu_available, is_xpu_available, is_peft_available

if is_peft_available():
    from peft import (
        PeftConfig,
        PeftModel,
        PeftModelForCausalLM,
        PeftModelForSeq2SeqLM,
        PromptLearningConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )





class AutoModelForSeq2SeqLMWithValueHead4PrefixTuning(AutoModelForSeq2SeqLMWithValueHead):
    r"""
    A seq2seq model with a value head in addition to the language model head.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained` and `push_to_hub` and also provides some additional
    functionalities such as `generate`.

    Args:
        pretrained_model (`transformers.PreTrainedModel`):
            The model to wrap. It should be a causal language model such as GPT2.
            or any model mapped inside the `AutoModelForSeq2SeqLM` class.
        kwargs:
            Additional keyword arguments passed along to the `ValueHead` class.
    """

    def __init__(self, pretrained_model, custom_arg=None, **kwargs):
       
        self.custom_arg = custom_arg

        super().__init__(pretrained_model, **kwargs)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model from `transformers`. The
        pretrained model is loaded using the `from_pretrained` method of the
        `transformers.PreTrainedModel` class. The arguments that are specific to the
        `transformers.PreTrainedModel` class are passed along this method and filtered
        out from the `kwargs` argument.


        Args:
            pretrained_model_name_or_path (`str` or `transformers.PreTrainedModel`):
                The path to the pretrained model or its name.
            *model_args (`list`, *optional*)):
                Additional positional arguments passed along to the underlying model's
                `from_pretrained` method.
            **kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's
                `from_pretrained` method. We also pre-process the kwargs to extract
                the arguments that are specific to the `transformers.PreTrainedModel`
                class and the arguments that are specific to trl models. The kwargs
                also support `prepare_model_for_kbit_training` arguments from
                `peft` library.
        """
        if kwargs is not None:
            peft_config = kwargs.pop("peft_config", None)
            reward_adapter = kwargs.pop("reward_adapter", None)
            reward_adapter_name = kwargs.pop("reward_adapter_name", "reward_adapter")
            is_trainable = kwargs.pop("is_trainable", False)
            trl_model_args, pretrained_kwargs, peft_quantization_kwargs = cls._split_kwargs(kwargs)
            token = pretrained_kwargs.get("token", None)
        else:
            peft_config = None
            is_trainable = False
            trl_model_args = {}
            pretrained_kwargs = {}
            peft_quantization_kwargs = {}
            token = None

        if reward_adapter is not None and not isinstance(reward_adapter, str):
            raise ValueError(
                "The `reward_adapter` argument should be a string representing the name of local path or the Hub id to the Reward Modeling adapter."
            )

        is_peft_model = False

        current_device = cls._get_current_device()
        if isinstance(pretrained_model_name_or_path, str):
            is_loaded_in_8bit = pretrained_kwargs["load_in_8bit"] if "load_in_8bit" in pretrained_kwargs else False
            is_loaded_in_4bit = pretrained_kwargs["load_in_4bit"] if "load_in_4bit" in pretrained_kwargs else False
        else:
            is_loaded_in_8bit = getattr(pretrained_model_name_or_path, "is_loaded_in_8bit", False)
            is_loaded_in_4bit = getattr(pretrained_model_name_or_path, "is_loaded_in_4bit", False)

        if (is_loaded_in_8bit or is_loaded_in_4bit) and "device_map" not in pretrained_kwargs:

            logging.warning(
                "The `device_map` argument is not provided. We will override the device_map argument."
                " to set the entire"
                " model on the current device. If you want to set the model on multiple devices, please provide"
                " a custom `device_map` argument."
            )
            pretrained_kwargs["device_map"] = {"": current_device}

        if is_peft_available() and peft_config is not None and not isinstance(peft_config, PeftConfig):
            raise ValueError("The `peft_config` argument should be an instance of `peft.PeftConfig` class.")


        if isinstance(pretrained_model_name_or_path, str):
            if is_peft_available():
                try:

                    remote_adapter_config = hf_hub_download(
                        pretrained_model_name_or_path,
                        "adapter_config.json",
                        token=token,
                    )
                except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
                    remote_adapter_config = None
            else:
                remote_adapter_config = None

            local_adapter_present = os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json"))

            if (local_adapter_present or remote_adapter_config is not None) and is_peft_available():
                if peft_config is not None:
                    logging.warning(
                        "`peft_config` argument ignored since a peft config file was found in "
                        f"{pretrained_model_name_or_path}"
                    )


                if local_adapter_present:
                    trained_adapter_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)
                else:
                    remote_adapter_dir = os.path.dirname(remote_adapter_config)
                    trained_adapter_config = PeftConfig.from_pretrained(remote_adapter_dir)


                pretrained_model = cls.transformers_parent_class.from_pretrained(
                    trained_adapter_config.base_model_name_or_path, *model_args, **pretrained_kwargs
                )


                pretrained_model = PeftModel.from_pretrained(
                    pretrained_model, pretrained_model_name_or_path, is_trainable=is_trainable
                )
                logging.info("Trained peft adapter loaded")
            else:
                pretrained_model = cls.transformers_parent_class.from_pretrained(
                    pretrained_model_name_or_path, *model_args, **pretrained_kwargs
                )

                if peft_config is not None:

                    if is_loaded_in_8bit or is_loaded_in_4bit:
                        pretrained_model = prepare_model_for_kbit_training(
                            pretrained_model,
                            **peft_quantization_kwargs,
                        )
                    pretrained_model = get_peft_model(pretrained_model, peft_config)
                    logging.info("peft adapter initialised")

        elif isinstance(pretrained_model_name_or_path, cls.supported_pretrained_model_architectures):
            pretrained_model = pretrained_model_name_or_path
            
            if peft_config is not None and isinstance(pretrained_model, PreTrainedModel):

                if is_loaded_in_8bit or is_loaded_in_4bit:
                    pretrained_model = prepare_model_for_kbit_training(
                        pretrained_model,
                        **peft_quantization_kwargs,
                    )
                pretrained_model = get_peft_model(pretrained_model, peft_config)
                logging.info("peft adapter initialised")
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string or a PreTrainedModel, "
                f"but is {type(pretrained_model_name_or_path)}"
            )

        if is_peft_available():
            if isinstance(pretrained_model, PeftModel):
                is_peft_model = True


        if not is_peft_model and reward_adapter is not None:
            raise ValueError("reward_adapter can only be used with a PeftModel. ")
        elif is_peft_model and reward_adapter is not None:
            score_module = cls.add_and_load_reward_modeling_adapter(
                pretrained_model, reward_adapter, reward_adapter_name, token=token
            )
            multi_adapter_args = {
                "score_module": score_module,
                "supports_rm_adapter": True,
                "rm_adapter_name": reward_adapter_name,
            }
        else:
            multi_adapter_args = {"supports_rm_adapter": False}


        model = cls(pretrained_model, **multi_adapter_args, **trl_model_args)


        is_resuming_training = True
        if isinstance(pretrained_model_name_or_path, str):
            safe_filename = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")

            sharded_index_filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin.index.json")
            safe_sharded_index_filename = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
            is_sharded = False
            use_safe = os.path.exists(safe_filename)

            if not (os.path.exists(filename) or os.path.exists(safe_filename)):

                filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(
                    pretrained_model,
                    pretrained_model_name_or_path,
                    sharded_index_filename,
                    token=token,
                )

                if filename is None and files_to_download is None:
                    safe_filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(
                        pretrained_model,
                        pretrained_model_name_or_path,
                        safe_sharded_index_filename,
                        token=token,
                        model_name="model.safetensors",
                        model_index_name="model.safetensors.index.json",
                    )
                    use_safe = True
                else:
                    use_safe = False

            loading_func = safe_load_file if use_safe else torch.load
            load_kwargs = {} if use_safe else {"map_location": "cpu"}

            if is_resuming_training:
                if is_sharded:

                    state_dict = {}

                    for shard_file in files_to_download:
                        filename = hf_hub_download(
                            pretrained_model_name_or_path,
                            shard_file,
                            token=token,
                        )
                        state_dict.update(loading_func(filename, **load_kwargs))
                else:
                    state_dict = loading_func(filename if not use_safe else safe_filename, **load_kwargs)

        else:
            state_dict = pretrained_model_name_or_path.state_dict()

        model.is_peft_model = is_peft_model
        model.current_device = current_device

        if is_resuming_training:
            model.post_init(state_dict=state_dict)

        return model

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["past_key_values"] = past_key_values
        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden_state = base_model_output.decoder_hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        value = self.v_head(last_hidden_state).squeeze(-1)

        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)

    def enable_input_require_grads(self):
        self.pretrained_model.enable_input_require_grads()
    
    
    def generate(self, *args, **kwargs):
        r"""
        We call `generate` on the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)