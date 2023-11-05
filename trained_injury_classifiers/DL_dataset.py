import argparse
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class dataset(Dataset):
    def __init__(self, ids, attn, type_ids, truth):
        self.ids = ids
        self.attn = attn
        self.type_ids = type_ids
        self.truth = truth

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.truth is not None:
            return self.ids[i], self.attn[i], self.type_ids[i], self.truth[i]
        else:
            return (self.ids[i], self.attn[i], self.type_ids[i])


def return_DL_types(args):
    unflatten_all_case = []
    all_case = []
    all_type = []
    small_type = []
    with open(args.train_file, "r") as f:
        freader = csv.reader(f, delimiter=",")
        for case in freader:
            unflatten_all_case.append(case[2].split(";"))
        for unflatten in unflatten_all_case[1:]:
            for x in unflatten:
                all_case.append(x)
        for cat in all_case:
            if not cat in all_type:
                all_type.append(cat)
        for tpe in all_type:
            if all_case.count(tpe) < 40:
                small_type.append(tpe)

    # 57 classes in total
    rule_based_type = []
    for tpe in small_type:
        if tpe not in [
            "Single Wound",
            "Multiple Wounds",
            "Rare Internal Injuries",
            "Other Damages",
            "Unspecified",
            "Unspecified Muscle Injuries",
        ]:
            rule_based_type.append(tpe)
    rule_based_type.append("Sleeping Disorder")
    rule_based_type.append("Kidney")
    rule_based_type.append("Gunshot")
    rule_based_type.append("Arthritis")
    rule_based_type.append("Death")
    rule_based_type.append("Survival Action")
    rule_based_type.append("Single Wound/Multiple Wounds",)

    # 35
    DL_based_type = []
    for tpe in all_type:
        if tpe not in rule_based_type and tpe not in [
            "Other Damages",
            "Unspecified",
            "Rare Internal Injuries",
            "Single Wound",
            "Multiple Wounds",
        ]:
            DL_based_type.append(tpe)

    # 35, 96
    # return DL_based_type, all_type
    return DL_based_type, all_type


def tokenize_descriptions(args, all_type, tokenizer):
    # train data
    injury_description = []
    with open(args.train_file, "r") as f:
        freader = csv.reader(f, delimiter=",")
        for case in freader:
            injury_description.append(case[0])
        injury_description = injury_description[1:]

    type_des = {}
    for tpe in all_type:
        type_des[tpe] = []
    with open(args.train_file, "r") as f:
        freader = csv.reader(f, delimiter=",")
        for case in freader:
            for tpe in all_type:
                if tpe in case[2].split(";"):
                    type_des[tpe].append(case[0])

    # test data
    test_data = []
    type_des_test = {}
    for tpe in all_type:
        type_des_test[tpe] = []
    with open(args.test_file, "r") as f:
        freader = csv.reader(f, delimiter=",")
        for case in freader:
            test_data.append(case[0])
            for tpe in all_type:
                if tpe in case[1].split(";"):
                    type_des_test[tpe].append(case[0])
    test_data = test_data[1:]

    return (
        injury_description,
        test_data,
        type_des,
        type_des_test,
    )


def generate_train_dataset(args, injury_description, type_des, tokenizer):
    assert args.injtype in type_des.keys()

    tokenized_label = tokenizer(args.injtype)
    tokenized_label_ids = torch.tensor(tokenized_label["input_ids"])
    tokenized_label_attn = torch.tensor(tokenized_label["attention_mask"])
    tokenized_label_type_ids = torch.tensor(tokenized_label["token_type_ids"])

    length_train = len(injury_description)

    # create train dataloader
    train_truth = type_des[args.injtype]
    binary_truth = torch.zeros(length_train)
    for i, injury in enumerate(injury_description):
        if injury in train_truth:
            binary_truth[i] = 1

    # positive sampling
    proportion = len(train_truth) / length_train
    if 0.4 <= proportion <= 0.5:
        print("the proportion of positive samples are acceptable")
        tokenized_injury_description = tokenizer(injury_description, padding=True)
        tokenized_injury_description_ids = torch.tensor(
            tokenized_injury_description["input_ids"]
        )
        tokenized_injury_description_attn = torch.tensor(
            tokenized_injury_description["attention_mask"]
        )
        tokenized_injury_description_type_ids = torch.ones_like(
            torch.tensor(tokenized_injury_description["attention_mask"])
        )
    else:
        required_proportion = random.uniform(0.4, 0.5)
        print(
            "the proportion of positive samples are not acceptable, need positive sampling"
        )
        need = int(
            (required_proportion * length_train - len(train_truth))
            / (1 - required_proportion)
        )
        new_proportion = (len(train_truth) + need) / (length_train + need)
        assert 0.4 <= new_proportion <= 0.5
        positive_indexes = random.choices(
            list(range(len(train_truth))), weights=None, cum_weights=None, k=need
        )

        indexes = random.choices(
            list(range(length_train)), weights=None, cum_weights=None, k=need
        )
        new_injuries = []
        for i, j in zip(positive_indexes, indexes):
            new_injuries.append(train_truth[i] + "; " + injury_description[j])
            binary_truth = torch.cat((binary_truth, torch.ones(1)))
        injury_description = injury_description + new_injuries

        tokenized_injury_description = tokenizer(injury_description, padding=True)
        tokenized_injury_description_ids = torch.tensor(
            tokenized_injury_description["input_ids"]
        )
        tokenized_injury_description_attn = torch.tensor(
            tokenized_injury_description["attention_mask"]
        )
        tokenized_injury_description_type_ids = torch.ones_like(
            torch.tensor(tokenized_injury_description["token_type_ids"])
        )

    tokenized_concat_description_ids = torch.cat(
        [
            tokenized_label_ids.unsqueeze(0).repeat(len(injury_description), 1),
            tokenized_injury_description_ids,
        ],
        dim=1,
    )
    tokenized_concat_description_attn = torch.cat(
        [
            tokenized_label_attn.unsqueeze(0).repeat(len(injury_description), 1),
            tokenized_injury_description_attn,
        ],
        dim=1,
    )
    tokenized_concat_description_type_ids = torch.cat(
        [
            tokenized_label_type_ids.unsqueeze(0).repeat(len(injury_description), 1),
            tokenized_injury_description_type_ids,
        ],
        dim=1,
    )

    train_dataset = dataset(
        tokenized_concat_description_ids,
        tokenized_concat_description_attn,
        tokenized_concat_description_type_ids,
        binary_truth,
    )

    return train_dataset


def generate_test_dataset(args, test_data, type_des, type_des_test, tokenizer):
    assert args.injtype in type_des.keys()

    tokenized_label = tokenizer(args.injtype)
    tokenized_label_ids = torch.tensor(tokenized_label["input_ids"])
    tokenized_label_attn = torch.tensor(tokenized_label["attention_mask"])
    tokenized_label_type_ids = torch.tensor(tokenized_label["token_type_ids"])

    test_truth = type_des_test[args.injtype]
    length_test = len(test_data)

    binary_test_truth = torch.zeros(length_test)
    for i, injury in enumerate(test_data):
        if injury in test_truth:
            binary_test_truth[i] = 1

    tokenized_injury_description_test = tokenizer(test_data, padding=True)
    tokenized_injury_description_test_ids = torch.tensor(
        tokenized_injury_description_test["input_ids"]
    )
    tokenized_injury_description_test_attn = torch.tensor(
        tokenized_injury_description_test["attention_mask"]
    )
    tokenized_injury_description_test_type_ids = torch.ones_like(
        torch.tensor(tokenized_injury_description_test["token_type_ids"])
    )

    tokenized_concat_description_test_ids = torch.cat(
        [
            tokenized_label_ids.unsqueeze(0).repeat(len(test_data), 1),
            tokenized_injury_description_test_ids,
        ],
        dim=1,
    )
    tokenized_concat_description_test_attn = torch.cat(
        [
            tokenized_label_attn.unsqueeze(0).repeat(len(test_data), 1),
            tokenized_injury_description_test_attn,
        ],
        dim=1,
    )
    tokenized_concat_description_test_type_ids = torch.cat(
        [
            tokenized_label_type_ids.unsqueeze(0).repeat(len(test_data), 1),
            tokenized_injury_description_test_type_ids,
        ],
        dim=1,
    )

    test_dataset = dataset(
        tokenized_concat_description_test_ids,
        tokenized_concat_description_test_attn,
        tokenized_concat_description_test_type_ids,
        binary_test_truth,
    )

    return test_dataset


def generate_prediction_dataset(args, test_data, type_des, tokenizer):
    assert args.injtype in type_des.keys()

    tokenized_label = tokenizer(args.injtype)
    tokenized_label_ids = torch.tensor(tokenized_label["input_ids"])
    tokenized_label_attn = torch.tensor(tokenized_label["attention_mask"])
    tokenized_label_type_ids = torch.tensor(tokenized_label["token_type_ids"])

    length_test = len(test_data)

    tokenized_injury_description_test = tokenizer(test_data, padding=True)
    tokenized_injury_description_test_ids = torch.tensor(
        tokenized_injury_description_test["input_ids"]
    )
    tokenized_injury_description_test_attn = torch.tensor(
        tokenized_injury_description_test["attention_mask"]
    )
    tokenized_injury_description_test_type_ids = torch.ones_like(
        torch.tensor(tokenized_injury_description_test["token_type_ids"])
    )

    tokenized_concat_description_test_ids = torch.cat(
        [
            tokenized_label_ids.unsqueeze(0).repeat(len(test_data), 1),
            tokenized_injury_description_test_ids,
        ],
        dim=1,
    )
    tokenized_concat_description_test_attn = torch.cat(
        [
            tokenized_label_attn.unsqueeze(0).repeat(len(test_data), 1),
            tokenized_injury_description_test_attn,
        ],
        dim=1,
    )
    tokenized_concat_description_test_type_ids = torch.cat(
        [
            tokenized_label_type_ids.unsqueeze(0).repeat(len(test_data), 1),
            tokenized_injury_description_test_type_ids,
        ],
        dim=1,
    )

    test_dataset = dataset(
        tokenized_concat_description_test_ids,
        tokenized_concat_description_test_attn,
        tokenized_concat_description_test_type_ids,
        None,
    )

    return test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--injtype", type=str, default="Head, Skull")
    parser.add_argument(
        "--train_file", type=str, default="head_skull_training_data.csv"
    )
    parser.add_argument("--test_file", type=str, default="test_data.csv")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", local_files_only=True
    )

    DL_based_type, all_type = return_DL_types(args)

    (injury_description, test_data, type_des, type_des_test,) = tokenize_descriptions(
        args, all_type, tokenizer
    )

    assert args.injtype in DL_based_type

    train_dataset = generate_train_dataset(
        args, injury_description, type_des, tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for i in train_loader:
        print(i)
