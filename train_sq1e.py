#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer

from peft import LoraConfig, TaskType, get_peft_model
# SenQnn related import
from sq1e.sen_qnn_ext import Qmodel_prep, senqnn_config_init, patch_torch_bmm
from torch.utils.tensorboard import SummaryWriter

from sq1e.qtrainer import Qtrainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

logging.basicConfig(level=logging.DEBUG)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    apply_lora: bool = field(
        default=False, metadata={"help": "Whether to train with lora"}
    )
    lora_r: int = field( default=8, metadata={"help": "rank of lora matrix"})
    lora_alpha: int = field( default=16, metadata={"help": "lora scaling factor"})
    dropout_prob_lora: float = field( default=0.1, metadata={"help": "rank dropout"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
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


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # --- added by Charlie for sq1e, use conventional way to add args ---
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
    # parser.add_argument('--Qskip_layer_name', action='append', help='skip this layer in quantization') 
    # ------------------------------------------------------
    model_args, data_args, training_args, sq_args = parser.parse_args_into_dataclasses()
    training_args.report_to=[]

    training_args.deepspeed = None
    training_args.hf_deepspeed_config = None
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torchscript=True
    )
    # model.to(torch.device('cuda'))

    model_args, data_args, training_args, sq_args = parser.parse_args_into_dataclasses()
    training_args.report_to=[]

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
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

    if sq_args.teacher_model:
        teacher_model = copy.deepcopy(model)
        checkpoint = torch.load(sq_args.teacher_model)
        teacher_model.load_state_dict(checkpoint)
        # teacher_model.eval()
        del checkpoint
    else:
        teacher_model = None

    model.to(torch.device('cuda'))
    # if teacher_model:
    #     teacher_model.to(torch.device('cuda'))

    if sq_args.lora_initialization_checkpoint:
        lora_initialization_checkpoint = torch.load(sq_args.lora_initialization_checkpoint, map_location='cpu')
    else:
        lora_initialization_checkpoint = None

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

    trainable_params = []
    if model_args.apply_lora:
        trainable_params.append('lora')
        if sq_args.nbits_a < 32:
            trainable_params.append("clip_val")

    # ----- added for sq1e -----
    sqcfg = senqnn_config_init(sq_args) # we added/parsed our args in sq_args
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
                    if 'clip_val' in k: self.tb_writer.add_scalar(k, v.item(), state.global_step)
                # YL: Hack to add pact_a_lr to tb
                # self.tb_writer.add_scalar('pact_a_lr', kwargs.get('optimizer').param_groups[2]['lr'], state.global_step) 
            return super().on_log(args, state, control, logs, **kwargs)
    # --------------------------

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Qtrainer(model=model, tokenizer=tokenizer, args=training_args,
                       callbacks=[sq1eCallback(sqcfg), sq1eTBcallback(tb_writer)], #https://huggingface.co/docs/transformers/main_classes/callback
                       teacher_model=teacher_model,
                       kd_ratio = sq_args.kd_ratio,
                       lora_initialization_checkpoint=lora_initialization_checkpoint, #may initilize lora weights from a higher precision checkpoint
                       **data_module)
    # --- added for sq1e ---
    trainer.sqcfg = sqcfg
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint, trainable_params=trainable_params)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
