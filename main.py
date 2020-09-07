import code

from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Union
from enum import Enum
import time
import random
import pickle

import pandas as pd
import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, PreTrainedTokenizer
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from modeling import DualRobertaForDotProduct

""" 创建一个加载Recommendation dataset数据集的库
"""
class RecProcessor(DataProcessor):
    """Processor for the Recommendation data set (GLUE version)."""

    def __init__(self, user_reviews_file, item_reviews_file):
        self.user_reviews = {}
        self.item_reviews = {}
        with open(user_reviews_file) as fin:
            for line in fin:
                line = line.strip().split("\t")
                self.user_reviews[line[0]] = line[1:]
        with open(item_reviews_file) as fin:
            for line in fin:
                line = line.strip().split("\t")
                self.item_reviews[line[0]] = line[1:]

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "valid.csv"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "test.csv"), "test")

    def _create_examples(self, filename, set_type):
        """Creates examples for the training, dev and test sets."""

        data = pd.read_csv(filename)
        user_review_examples = []
        item_review_examples = []
        for i in range(data.shape[0]):
            guid = "%s-%s" % (set_type, i)
            row = data.iloc[i]
            user_id = str(row["user_id"])
            item_id = str(row["item_id"])
            label = float(row["ratings"])
            rating = row["ratings"]
            user_reviews = self.user_reviews[user_id] if user_id in self.user_reviews else ["N/A"]
            item_reviews = self.item_reviews[item_id] if item_id in self.user_reviews else ["N/A"]
            random.shuffle(user_reviews)
            random.shuffle(item_reviews)
            user_reviews = " [SEP] ".join(user_reviews)
            item_reviews = " [SEP] ".join(item_reviews)
            user_review_examples.append(InputExample(guid=guid, text_a=user_reviews, label=label))
            item_review_examples.append(InputExample(guid=guid, text_a=item_reviews, label=label))

        return user_review_examples, item_review_examples

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on"})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files (or other data files) for the task."}
    )
    user_reviews_file: str = field(
        metadata={"help": "The file containing all user reviews"}
    )
    item_reviews_file: str = field(
        metadata={"help": "The file containing all item reviews"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


""" 在这个function中，我们会把文本数据转化为可以传入BERT模型的index, mask等输入
"""
def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    batch_encoding = tokenizer(
        [example.text_a for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        # https://github.com/huggingface/transformers/blob/master/src/transformers/data/processors/utils.py#L56
        # InputFeatures当中包含了input_ids, attention_mask, token_type_ids和label四个部分
        # feature = InputFeatures(**inputs)
        features.append(inputs)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


""" RecDataset这个库继承了PyTorch自带的Dataset库。转换成dataloader之后可以用来做训练和测试
"""
class RecDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: DataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = RecProcessor(args.user_reviews_file, args.item_reviews_file)
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.

        logger.info(f"Creating features from dataset file at {args.data_dir}")

        if mode == Split.dev:
            user_review_examples, item_review_examples = self.processor.get_dev_examples(args.data_dir)
            pickle_path = os.path.join(args.data_dir, "features.dev.pkl")
        elif mode == Split.test:
            user_review_examples, item_review_examples = self.processor.get_test_examples(args.data_dir)
            pickle_path = os.path.join(args.data_dir, "features.test.pkl")
        else:
            user_review_examples, item_review_examples = self.processor.get_train_examples(args.data_dir)
            pickle_path = os.path.join(args.data_dir, "features.train.pkl")

        if os.path.exists(pickle_path):
            logger.info("*** Loading features from the pickle file ***")
            self.user_review_features, self.item_review_features = pickle.load(open(pickle_path, "rb"))
        else:
            logger.info("*** Creating the feature pickle file ***")
            self.user_review_features = convert_examples_to_features(
                user_review_examples,
                tokenizer,
                max_length=args.max_seq_length,
            )
            self.item_review_features = convert_examples_to_features(
                item_review_examples,
                tokenizer,
                max_length=args.max_seq_length,
            )
            pickle.dump([self.user_review_features, self.item_review_features], open(pickle_path, "wb"))
        


        assert len(self.user_review_features) == len(self.item_review_features)

        self.features = []
        for i in range(len(self.user_review_features)):
            feature = {}
            for k, v in self.user_review_features[i].items():
                feature["user_" + k] = v
            for k, v in self.item_review_features[i].items():
                feature["item_" + k] = v
            feature["labels"] = user_review_examples[i].label
            self.features.append(feature)

        # code.interact(local=locals())

        start = time.time()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )





def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    output_mode = "regression"

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # tokenizer，用来做分词等数据预处理工作
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    train_dataset = RecDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
    # num_labels = len(train_dataset.get_labels())

    # config 包含了模型的基本参数设定
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    # 加载模型
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # ) #.cuda()
    model = DualRobertaForDotProduct.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )


    # Get datasets
    eval_dataset = (
        RecDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        RecDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    def mse(preds, labels):
        return ((preds - labels)*(preds - labels)).mean()

    def compute_metrics_fn(p: EvalPrediction):
        preds = p.predictions
        return {"Rec": mse(preds, p.label_ids)}


    # Initialize our Trainer
    # 模型训练代码，非常值得一读 https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L134
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
    )


    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = compute_metrics_fn
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        writer.write("%d\t%3.3f\n" % (index, item))
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
