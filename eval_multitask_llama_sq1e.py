#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Dict
import time
import glob

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from torch.utils.data import DataLoader
from sq1e.sen_qnn_ext import Qmodel_prep, senqnn_config_init, patch_torch_bmm, get_act_scales, \
                             DQ_LLM, Qmodel_calib
# from sq1e.sen_qnn_utils import get_gpu_memory_usage
from lm_eval import tasks, evaluator, utils
from lm_eval.models import get_model
from torch.utils.tensorboard import SummaryWriter

from peft import LoraConfig, TaskType, get_peft_model

# SenQnn related import
from sq1e.sen_qnn_ext import Qmodel_prep, senqnn_config_init, patch_torch_bmm
from torch.utils.tensorboard import SummaryWriter

from sq1e.qtrainer import Qtrainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    apply_lora: bool = field(
        default=False, metadata={"help": "Whether to train with lora"}
    )
    lora_r: int = field( default=8, metadata={"help": "rank of lora matrix"})
    lora_alpha: int = field( default=16, metadata={"help": "lora scaling factor"})
    dropout_prob_lora: float = field( default=0.1, metadata={"help": "rank dropout"})

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fake_training: bool = field(
        default=False, metadata={"help": "set model to evaluation mode to still run sq1e but same memory"}
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    #####for PTQ
    parser.add_argument('--gpu1_eval',  action='store_true', help='use 1gpu for evaluation')
    parser.add_argument('--cpu_eval',  action='store_true', help='use cpu for evaluation')
    parser.add_argument('--do_multi_eval',  action='store_true', help='do eval on multiple downstream tasks')
    parser.add_argument('--eval_only_ckpt_dir', type=str, default='', help='directory with checkpoint for evaluation (only run eval, no PTQ)')
    parser.add_argument('--isTransformers',  action='store_true', help='ptq transformers')
    parser.add_argument('--hasEncoder',  action='store_true', help='PTQ: transformer model with encoder')
    parser.add_argument('--hasDecoder',  action='store_true', help='PTQ: transformer model with decoder')
    parser.add_argument('--decoder_arch',  action='store_true', help='search q candidate for decoder arch to avoid skipping to many layers')
    parser.add_argument('--save_ckpt',  action='store_true', help='save qmodel ckpt after tuning')
    parser.add_argument('--sq1e_dtype', default=None, type=str, choices=['float32', 'float16', 'bfloat16'], help='change Qmodel data type')
    # multi-task evaluation
    parser.add_argument('--tasks', nargs='+', default=[], help='multi-task evaluation for zero-shot or 5-shot eval')
    parser.add_argument('--task_limit', action='store_true', help='Whether to limit the samples for the evaluation tasks')
    parser.add_argument('--task_limit_percent', default=0.2, type=float, help='percentage of evaluation tasks')
    parser.add_argument('--num_fewshot', default=0, type=int, help='number of few-shot eval, default is zero-shot')
    parser.add_argument('--task_mmlu', action='store_true', help='Whether to also evaluate on the mmlu task.')
    parser.add_argument('--layer_reset_quantizer', nargs='+', default=[], help='reset quantizers for layers with certain name strings')

    parser.add_argument('--nbits_w', default=32, type=int, help='weight precision')
    parser.add_argument('--nbits_a', default=32, type=int, help='activation precision')
    parser.add_argument('--nbits_w_qkv', default=32, type=int, help='weight precision for qkv layers')
    parser.add_argument('--nbits_a_qkv', default=32, type=int, help='weight precision for qkv layers')
    parser.add_argument('--nbits_bmm1', default=32, type=int, help='weight precision for bmm1')
    parser.add_argument('--nbits_bmm2', default=32, type=int, help='weight precision for bmm2')
    parser.add_argument('--qw_mode', type=str, default='sawb', help='weight quantization, pick from lpuq, sawb or dorefa') 
    parser.add_argument('--qa_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
    parser.add_argument('--qw_qkv_mode', type=str, default='sawb', help='activation quantization, pick from lpuq, lsq or qil') 
    parser.add_argument('--qa_qkv_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
    parser.add_argument('--bmm1_qm1_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
    parser.add_argument('--bmm1_qm2_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
    parser.add_argument('--bmm2_qm1_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
    parser.add_argument('--bmm2_qm2_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
    parser.add_argument('--pact_a_lr', default=0.01, type=float, help='clip val learning rate') 
    parser.add_argument('--pact_w_lr', default=0.01, type=float, help='clip val learning rate') 
    parser.add_argument('--a_clip_val', type=float, default=6.0, help='clip_val initial value')
    parser.add_argument('--a_clip_valn', type=float, default=0.0, help='clip_valn initial value, specifically for QIL')
    parser.add_argument('--w_clip_val', type=float, default=1.0, help='positive weight clip_val initial value')   
    parser.add_argument('--w_clip_valn', type=float, default=-1.0, help='negative weight clip_val initial value')
    parser.add_argument('--pact_a_decay', default=5e-5, type=float, help='clip val for qil pruning clip decay') 
    parser.add_argument('--pact_w_decay', default=5e-5, type=float, help='clip val for W decay') 
    parser.add_argument('--align_zero',  action='store_true', help='set align_zero flags in W and A quantizers to True')
    parser.add_argument('--sentient_check',  action='store_true')
    parser.add_argument('--Qmodel_calibration',  default=0, type=int, help='Num of batches for Qmodel calibration')
    parser.add_argument('--Qmodel_calibration_new',  default=0, type=int, help='new method for calibration')
    parser.add_argument('--QKVsync',  action='store_true', help='synchronize clipvals of QKV layers')
    parser.add_argument('--clip_val_asst_percentile', nargs='+', type=float, default=(0.1,99.9), help='pecentile for clip_val initialization')
    parser.add_argument('--dropout_prob_attn', type=float, default=0.1, help='in hf3 we changed all dropout prob to 0.165')
    parser.add_argument('--dropout_prob_hid', type=float, default=0.1, help='in hf3 we changed all dropout prob to 0.165')
    parser.add_argument('--dropout_prob_emb', type=float, default=0.1, help='in hf3 we changed all dropout prob to 0.165')
    parser.add_argument('--plotSVG',  action='store_true', help='save computation graphs, needs graphviz/pygraphviz')
    parser.add_argument('--teacher_model', type=str, default='', help='teacher model to run distillation during QAT') 
    parser.add_argument('--kd_ratio', type=float, default=1.0, help='loss ratio for knowledge distillation')
    parser.add_argument('--lora_initialization_checkpoint', type=str, default='', help='initialize lora weights from a may be higher precision checkpoint') 
    # -----------------------------------------
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, senqnn_args = parser.parse_args_into_dataclasses()

    # do_eval_only = (
    #     senqnn_args.eval_only_ckpt_dir
    #     or (not training_args.do_train
    #         and not senqnn_args.do_llm_ptq
    #         and not senqnn_args.do_llm_dq)
    # )
    # if senqnn_args.QKVsync and do_eval_only:
    #     raise ValueError("QKVsync flag is not compatible with evaluation only")

    print(f"MODEL ARGS:\n{model_args}")
    print(f"DATA ARGS\n {data_args}")
    print(f"TRAINING ARGS:\n{training_args}")
    print(f"SENQNN ARGS:\n{senqnn_args}")

    try:
        local_rank = os.environ['LOCAL_RANK']
    except KeyError:
        local_rank = "0"
        TrainingArguments.place_model_on_device=False
    print(f'local rank is: {local_rank}')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # if training_args.should_log:
    #     # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    #     transformers.utils.logging.set_verbosity_info()
    fhandler = logging.FileHandler(filename = f'{training_args.output_dir}/results.txt')
    logger.addHandler(fhandler)

    # log_level = training_args.get_process_log_level()
    log_level = logging.DEBUG
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
    logger.info(raw_datasets)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    device = torch.device("cuda")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torchscript=True,
    )
    model.to(device)

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )


    # settings for LoRA
    if model_args.apply_lora:
        # target_modules = ['encoder.*query', 'encoder.*key', 'encoder.*value', 'encoder.*dense', "classifier.dense"]
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', 'lm_head']
        # target_modules = ['lm_head']
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                 inference_mode=False,
                                 r=model_args.lora_r,
                                 lora_alpha=model_args.lora_alpha,
                                 lora_dropout=model_args.dropout_prob_lora,
                                 target_modules=target_modules)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # ----- added for sq1e -----
    sqcfg = senqnn_config_init(senqnn_args) # we added/parsed our args in sq_args
    tb_writer=SummaryWriter(log_dir=f"{training_args.output_dir}/runs")
    sqcfg['dropout_prob_lora'] = model_args.dropout_prob_lora

    # class sq1eCallback(transformers.integrations.TrainerCallback):
    class sq1eCallback(transformers.trainer_callback.TrainerCallback):
        "initialize sq1e at the beginning of training, see trainer.py line 1358"
    # model, optimizer and ...etc are passed thru kwargs, 
    # see https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/trainer_callback.py#L159
    # NOTE there is another DefaultFlow TrainerCallback already, no need to run super().on_xxx_yyy() here
        def __init__(self, sqcfg):
            super().__init__()
            self.sqcfg = sqcfg

        def on_train_begin(self, args, state, control, **kwargs):
            if self.sqcfg['Qmodel_calibration'] > 0:
                Qmodel_prep(kwargs.get('model'), 
                            kwargs.get('train_dataloader'), self.sqcfg, 
                            kwargs.get('optimizer'), 
                            scheduler = args.lr_scheduler,
                            prefwdproc=lambda datamb: (datamb['input_ids'].to(args.device),), 
                            save_fname=''.join((args.output_dir, '/model', '.hf4')))
            else:
                Qmodel_prep(kwargs.get('model'), 
                            kwargs.get('train_dataloader'), self.sqcfg, 
                            kwargs.get('optimizer'), 
                            scheduler = args.lr_scheduler,
                            save_fname=''.join((args.output_dir, '/model', '.hf4')))

                            # To addl scheduler for pact value
                            # scheduler = args.lr_scheduler,
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == self.sqcfg['Qmodel_calibration_new']:
                print( {k:v for k,v in kwargs.get('model').named_parameters() if 'clip_val' in k} )

    class sq1eTBcallback(transformers.integrations.TensorBoardCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            # record clip_vals
            if self.tb_writer:
                for k, v in kwargs.get('model').named_parameters():
                    if 'clip_val' in k: self.tb_writer.add_scalar(k, v, state.global_step)
                # YL: Hack to add pact_a_lr to tb
                # self.tb_writer.add_scalar('pact_a_lr', kwargs.get('optimizer').param_groups[2]['lr'], state.global_step) 
            return super().on_log(args, state, control, logs, **kwargs)
    # --------------------------

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    print(f"MODEL DEVICE: {model.device}")
    print("MODEL PARAMS:")
    print("\n" + "\n".join(f"{k:70} {v.numel():10}     {list(v.size())}" for k,v in model.named_parameters()))
    print(f"TOTAL: {sum([v.numel() for k,v in model.named_parameters()]):,}\n")

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        logger.info(f"Grouping texts in chunks of {block_size}")
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval or senqnn_args.gpu1_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # loading checkpoint for evaluation
    if senqnn_args.eval_only_ckpt_dir:
        logger.debug(f'Loading model parameters from `{senqnn_args.eval_only_ckpt_dir}`')
        ckpts = glob.glob(senqnn_args.eval_only_ckpt_dir + '/pytorch_model*.bin') # capture all shards
        if len(ckpts) == 0:
            raise ValueError(f"No pytorch_model*.bin checkpoint found at `{senqnn_args.eval_only_ckpt_dir}`")
        state_dict = {}
        for ckpt in ckpts:
            logger.debug(f"Loading checkpoint {ckpt}")
            state_dict = state_dict | torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state_dict, strict=False) 
        del ckpts

    # TrainingArguments.place_model_on_device=False # for PTQ_LLM
    if sqcfg['cpu_eval']:
        training_args.no_cuda = True

    trainer = Qtrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
        callbacks=[sq1eCallback(sqcfg), sq1eTBcallback(tb_writer)], #https://huggingface.co/docs/transformers/main_classes/callback
    )

    # Training
    if training_args.do_train:
        trainer.sqcfg = sqcfg
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint, fake_training=training_args.fake_training)
        # trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()

    # Evaluation
    time_do_eval_start = None
    if training_args.do_eval:
        time_do_eval_start = time.time()
        # if sqcfg['cpu_eval']:
            # trainer.args.device = torch.device("cpu")
        logger.info("*** Evaluate ***")
        # metrics = trainer.evaluate()

        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        # try:
        #     perplexity = math.exp(metrics["eval_loss"])
        # except OverflowError:
        #     perplexity = float("inf")
        # metrics["perplexity"] = perplexity

        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)
        # logger.info(f"Eval time on {data_args.dataset_name}: {time.time() - time_do_eval_start} s")
        # logger.info(f"Compute (model dtype): {model.dtype}")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if senqnn_args.do_multi_eval:
        if sqcfg['hasDecoder']:
            lm_eval_model_type = 'hf-seq2seq' if sqcfg['hasEncoder'] else 'hf-causal-experimental'
            # must disable torchscript, otherwise the model cannot return a dictionary
            # but lm_eval expects a dictionary with 'logits' keyword
            #
            # see configuration_utils.py
            #   PretrainedConfig.use_return_dict
            #       If torchscript is set, force `return_dict=False` to avoid jit errors
            #       return self.return_dict and not self.torchscript
            #
            # see huggingface.py:
            #   AutoSeq2SeqLM._loglikelihood_tokens
            #       log_softmaxes = F.log_softmax(outputs.logits, dim=-1)
            model.config.torchscript = False
            model.config.return_dict = True

            model.config.return_dict_in_generate = True # maybe not needed
            model.config.use_cache = False # maybe not needed
        else:
            raise ValueError("Encoder-only models not currently supported")

        model_wrapper_eval = get_model(lm_eval_model_type)(
            senqnn_args.eval_only_ckpt_dir,
            # model_args.model_name_or_path,
            model = model.cuda(),
            batch_size = training_args.per_device_eval_batch_size,
        )
        # model_wrapper_eval.model = model.cuda()
        print(model_wrapper_eval.model)

        # mem_used, mem_tot = get_gpu_memory_usage(device=0)
        # logger.info(f"CUDA memory usage: {mem_used} / {mem_tot} MiB")

        model_wrapper_eval.model.eval()

        task_limit_perc = senqnn_args.task_limit_percent if senqnn_args.task_limit else None

        time_multi_eval_start = time.time()
        #run 11 zero-shot or mmlu 5-shot
        if senqnn_args.task_mmlu:
            mmlu_tasks = ['hendrycksTest-abstract_algebra',
                          'hendrycksTest-anatomy', 'hendrycksTest-astronomy',
                          'hendrycksTest-business_ethics', 'hendrycksTest-clinical_knowledge',
                          'hendrycksTest-college_biology', 'hendrycksTest-college_chemistry',
                          'hendrycksTest-college_computer_science', 'hendrycksTest-college_mathematics',
                          'hendrycksTest-college_medicine', 'hendrycksTest-college_physics',
                          'hendrycksTest-computer_security', 'hendrycksTest-conceptual_physics',
                          'hendrycksTest-econometrics', 'hendrycksTest-electrical_engineering',
                          'hendrycksTest-elementary_mathematics', 'hendrycksTest-formal_logic',
                          'hendrycksTest-global_facts', 'hendrycksTest-high_school_biology',
                          'hendrycksTest-high_school_chemistry', 'hendrycksTest-high_school_computer_science',
                          'hendrycksTest-high_school_european_history', 'hendrycksTest-high_school_geography',
                          'hendrycksTest-high_school_government_and_politics', 'hendrycksTest-high_school_macroeconomics',
                          'hendrycksTest-high_school_mathematics', 'hendrycksTest-high_school_microeconomics',
                          'hendrycksTest-high_school_physics', 'hendrycksTest-high_school_psychology',
                          'hendrycksTest-high_school_statistics', 'hendrycksTest-high_school_us_history',
                          'hendrycksTest-high_school_world_history', 'hendrycksTest-human_aging',
                          'hendrycksTest-human_sexuality', 'hendrycksTest-international_law',
                          'hendrycksTest-jurisprudence', 'hendrycksTest-logical_fallacies',
                          'hendrycksTest-machine_learning', 'hendrycksTest-management',
                          'hendrycksTest-marketing', 'hendrycksTest-medical_genetics',
                          'hendrycksTest-miscellaneous', 'hendrycksTest-moral_disputes',
                          'hendrycksTest-moral_scenarios', 'hendrycksTest-nutrition',
                          'hendrycksTest-philosophy', 'hendrycksTest-prehistory',
                          'hendrycksTest-professional_accounting', 'hendrycksTest-professional_law',
                          'hendrycksTest-professional_medicine', 'hendrycksTest-professional_psychology',
                          'hendrycksTest-public_relations', 'hendrycksTest-security_studies',
                          'hendrycksTest-sociology', 'hendrycksTest-us_foreign_policy',
                          'hendrycksTest-virology', 'hendrycksTest-world_religions'
                          ]
            with patch_torch_bmm(sqcfg):
                # NOTE: set no_cache to True, otherwise directly load previous results
                mmlu_results = evaluator.simple_evaluate(
                    model=model_wrapper_eval,
                    model_args="",
                    tasks=mmlu_tasks,
                    num_fewshot=5,
                    batch_size=None, # only matters if model is built from str (not our case)
                    max_batch_size=None, # only matters if model is built from str (not our case)
                    device="cuda:0",
                    no_cache=True,
                    limit=task_limit_perc,
                    description_dict=None,
                    decontamination_ngrams_path=None,
                    check_integrity=False,
                    write_out=False,
                    output_base_path=training_args.output_dir,
                )

            print(evaluator.make_table(mmlu_results))
            mmlu_total = 0.0
            for task in mmlu_results['results']:
                if 'acc' in mmlu_results['results'][task].keys():
                    mmlu_total += mmlu_results['results'][task]['acc']
            mmlu_avg_acc = mmlu_total / len(mmlu_tasks)

            logger.info(evaluator.make_table(mmlu_results))
            logger.info(f'mmlu avg acc: {mmlu_avg_acc}')
            logger.info(f"Total MMLU time: {time.time() - time_multi_eval_start} s")

        # run other multitasks
        if senqnn_args.tasks:
            time_tasks_start = time.time()
            results = evaluator.simple_evaluate(
                model=model_wrapper_eval,
                model_args="",
                tasks=senqnn_args.tasks,
                num_fewshot=senqnn_args.num_fewshot,
                batch_size=None,
                max_batch_size=None,
                device="cuda:0",
                no_cache=True,
                limit=task_limit_perc,
                description_dict=None,
                decontamination_ngrams_path=None,
                check_integrity=False,
                write_out=False,
                output_base_path=training_args.output_dir,
            )

            print(evaluator.make_table(results))
            logger.info(f"Task versions: {results['versions']}")
            total = 0.0
            for task in results['results']:
                if 'acc' in results['results'][task].keys():
                    total += results['results'][task]['acc']
                if 'word_perplexity' in results['results'][task].keys():
                    total = results['results'][task]['word_perplexity']
            if len(senqnn_args.tasks) > 1:
                avg_acc = total / len(senqnn_args.tasks)
                print('Avg acc:', round(avg_acc, 4))

            logger.info(evaluator.make_table(results))
            if len(senqnn_args.tasks) > 1:
                logger.info(f'Avg acc: {avg_acc}')
            else:
                logger.info(f'word_perplexity: {total}')
            logger.info(f"Total tasks time: {time.time() - time_tasks_start} s")

        time_init = time_multi_eval_start if time_do_eval_start is None else time_do_eval_start
        logger.info(f"Total eval time: {time.time() - time_init} s")
        logger.info(f"Compute (model dtype): {model.dtype}")

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
    print("End of script")
