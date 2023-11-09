import argparse
import csv
import os
import random

import torch
from torch.utils.data import DataLoader
from transformers import (BertModel, BertTokenizer)
from transformers.optimization import get_linear_schedule_with_warmup
from DL_dataset import (generate_prediction_dataset, generate_test_dataset,
                        generate_train_dataset, return_DL_types,
                        tokenize_descriptions)
from DL_model import FeedForwardClassifier

PATH_TO_MODEL = ''

def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)


def construct_optimizer(args, model, num_train_examples):
    no_weight_decay = ["LayerNorm.weight", "bias"]
    optimized_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(np in n for np in no_weight_decay)
            ],
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(np in n for np in no_weight_decay)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    # Implements linear decay of the learning rate.
    # default to be AdamW based on tensorflow, AdamWeightDecayOptimizer
    # parameters are using default
    optimizer = torch.optim.Adam(optimized_parameters, lr=args.learning_rate)

    num_training_steps = int(
        args.epoch
        * num_train_examples
        / (args.batch_size * args.accumulate_gradient_steps)
    )
    num_warmup_steps = int(args.warm_up_proportion * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    return optimizer, scheduler


def create_suitable_string(tokenizer, current_string):
    need_string = tokenizer.decode(current_string)
    cls_n = 2
    cls_now = 0
    start_pos = 0
    end_pos = 0
    for pos in range(len(need_string)):
        if need_string[pos : pos + 5] == "[CLS]":
            cls_now += 1
            if cls_now == cls_n:
                start_pos = pos + 6
    for pos in range(start_pos, len(need_string)):
        if need_string[pos : pos + 5] == "[SEP]":
            end_pos = pos

    return need_string[start_pos:end_pos]


def main(args):
    device = torch.device("cpu")
    prediction_results_list = [f for f in os.listdir(args.output_path) if f.endswith('.csv')]
    if any([args.injtype in f for f in prediction_results_list]):
        print(args.output_path + ': ' + args.injtype + ' exists!')
        return

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", local_files_only=False
    )

    DL_based_type, all_type = return_DL_types(args)

    (injury_description, test_data, type_des, type_des_test,) = tokenize_descriptions(
        args, all_type, tokenizer
    )

    assert args.injtype in DL_based_type

    encoder = BertModel.from_pretrained("bert-base-uncased", local_files_only=False)
    classifier = FeedForwardClassifier(args, encoder).to(device)
    # zero the parameter gradients
    classifier.zero_grad()

    # if args.train:
    #     train_dataset = generate_train_dataset(
    #         args, injury_description, type_des, tokenizer
    #     )
    #     print(train_dataset[0])
    #
    #     train_loader = DataLoader(
    #         train_dataset, batch_size=args.batch_size, shuffle=True
    #     )
    #
    #     print("Finished loading training data")
    #
    #     test_dataset = generate_test_dataset(
    #         args, test_data, type_des, type_des_test, tokenizer
    #     )
    #
    #     test_loader = DataLoader(
    #         test_dataset, batch_size=args.batch_size, shuffle=False
    #     )
    #
    #     print("Finished loading test data")
    #
    #     wait = 0
    #
    #     print("Start training the classifier")
    #
    #     wait_step = args.wait_step
    #
    #     # dp = torch.cuda.device_count() > 1
    #
    #     optimizer, scheduler = construct_optimizer(args, classifier, len(train_dataset))
    #
    #     num_steps = 1
    #     running_loss = 0.0
    #     for epoch in range(args.epoch):  # loop over the dataset multiple times
    #         classifier.train()
    #         for i, data in enumerate(train_loader):
    #
    #             # forward + backward + optimize
    #             loss, outputs = classifier("train", data)
    #
    #             # print("this is the outputs {}".format(outputs))
    #             # print("this is the labels {}".format(labels))
    #
    #             loss.backward()
    #
    #             if i % args.accumulate_gradient_steps == 0:
    #                 torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.clip)
    #                 optimizer.step()
    #                 scheduler.step()
    #                 classifier.zero_grad()
    #                 num_steps += 1
    #
    #             # print statistics
    #             running_loss += loss.item()
    #
    #             if num_steps % 10 == 0:  # print every 10 mini-batches
    #                 print(
    #                     "In epoch {} batch {}, running loss is {}".format(
    #                         epoch, i, running_loss
    #                     )
    #                 )
    #                 running_loss = 0.0
    #
    #         classifier.eval()
    #         difference = 0.0
    #         best_accuracy = args.required_accuracy
    #         for i, data in enumerate(train_loader, 0):
    #             _, _, _, train_labels = data
    #             train_labels = train_labels.to(device)
    #             prediction = classifier("eval", data)
    #             difference += torch.sum(torch.abs((prediction - train_labels)))
    #         accuracy = (
    #             1 - difference / torch.tensor([len(injury_description)]).cuda()
    #         )* 100
    #
    #         print("In epoch {}, the training accuracy is {}".format(epoch, accuracy))
    #
    #         if accuracy > best_accuracy:
    #             torch.save(
    #                 classifier.state_dict(), PATH_TO_MODEL + args.embed_model + "-{}.pt".format(args.injtype)
    #             )
    #
    #             best_accuracy = accuracy
    #             print(
    #                 "In epoch {}, the best training accuracy is {}".format(
    #                     epoch, best_accuracy
    #                 )
    #             )
    #
    #             wait = 0
    #
    #             test_difference = 0
    #             for i, data in enumerate(test_loader, 0):
    #                 _, _, _, test_labels = data
    #                 test_labels = test_labels.cuda()
    #                 prediction = classifier("eval", data)
    #             test_difference += torch.sum(torch.abs((prediction - test_labels)))
    #             test_accuracy = (
    #                 1 - test_difference / torch.tensor([len(test_data)]).cuda()
    #             )* 100
    #             print(
    #                 "In epoch {}, the trained classifier for {} with test accuracy {}.".format(
    #                     epoch, args.injtype, test_accuracy
    #                 )
    #             )
    #         else:
    #             wait += 1
    #             if wait == wait_step:
    #                 break
    #
    #     print("finish epoch {} and the training accuracy is {}".format(epoch, accuracy))

    if args.predict and not args.all_models:
        classifier.load_state_dict(
            torch.load(r"C:\palm\trained_injury_classifiers" + r"\best-classifier-{}.pt".format(args.injtype), map_location = device)
        )
        classifier.eval()

        injury_descriptions = []
        with open(args.input_path + args.prediction_file, "r") as f:
            freader = csv.reader(f, delimiter=",")
            for case in freader:
                injury_descriptions.append(case[0])

        prediction_dataset = generate_prediction_dataset(
            args, injury_descriptions, type_des, tokenizer
        )
        prediction_loader = DataLoader(
            prediction_dataset, batch_size=args.batch_size, shuffle=False
        )

        description_result = []
        for i, data in enumerate(prediction_loader, 0):
            print(tokenizer.decode(data[0][0]))
            print(data)

            prediction = classifier("eval", data)
            for j in range(data[0].size(0)):
                need_string = create_suitable_string(tokenizer, data[0][j])
                print(need_string)
                if prediction[j] == 1:
                    description_result.append((need_string, args.injtype))
                else:
                    description_result.append((need_string, ""))

        with open(args.output_path + args.prediction_result, "w") as f:
            fieldnames = ["injury_description", "prediction_type"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for prediction in description_result:
                writer.writerow(
                    {
                        "injury_description": prediction[0],
                        "prediction_type": prediction[1],
                    }
                )

    if args.predict and args.all_models:
        all_injtypes = ['Abdomen', 
                        'Arm', 
                        'Brain', 
                        'Breast or Chest', 
                        'Burns', 
                        'Cognitive Problem', 
                        'Ear', 
                        'Eye', 
                        'Face', 
                        'Finger', 
                        'Foot', 
                        'General Emotional Problems', 
                        'Hand', 
                        'Head, Skull', 
                        'Heart', 
                        'Hip, Pelvis', 
                        'Infection', 
                        'Jaw', 
                        'Joint', 
                        'Leg', 
                        'Lip', 
                        'Lung', 
                        'Mouth or Teeth', 
                        'Neck or Back', 
                        'Nerve System', 
                        'Nose', 
                        'Ribs', 
                        'Sexual Dysfunction', 
                        'Shoulder', 
                        'Skin', 
                        'Soft Tissue', 
                        'Spine', 
                        'Stomach', 
                        'Unspecified Muscle Injuries', 
                        'Wrist']

        description_result = []

        injury_descriptions = []
        with open(args.input_path + args.prediction_file, "r") as f:
            freader = csv.reader(f, delimiter=",")
            for case in freader:
                injury_descriptions.append(case[0])



        for injtype in all_injtypes:

            classifier.load_state_dict(
                torch.load(r"\home\zhexuan_li" + r"\best-classifier-{}.pt".format(injtype), map_location = device)
            )
            classifier.eval()
            # print(description_result)
            args.injtype = injtype
            prediction_dataset = generate_prediction_dataset(
                args, injury_descriptions, type_des, tokenizer
            )

            prediction_loader = DataLoader(
                prediction_dataset, batch_size=args.batch_size, shuffle=False
            )

            if not description_result:
                for i, data in enumerate(prediction_loader, 0):
                    # print(tokenizer.decode(data[0][0]))
                    # print(data)
                    prediction = classifier("eval", data)
                    # print(prediction)
                    for j in range(len(prediction)):
                        need_string = create_suitable_string(tokenizer, data[0][j])
                        if prediction[j] == 1:
                            description_result.append((need_string, [injtype]))
                        else:
                            description_result.append((need_string, []))
            else:
                n = 0
                for i, data in enumerate(prediction_loader, 0):
                    # print(tokenizer.decode(data[0][0]))
                    prediction = classifier("eval", data)
                    for j in range(len(prediction)):
                        if prediction[j] == 1:
                            description_result[n][1].append(injtype)
                        n += 1
        with open(args.output_path + args.prediction_result, "w") as f:
            fieldnames = ["injury_description", "prediction_types"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for prediction in description_result:
                # print(prediction)
                writer.writerow(
                    {
                        "injury_description": prediction[0],
                        "prediction_types": prediction[1],
                    }
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, default="training_data.csv")
    parser.add_argument(
        "--test_file",
        type=str,
        default="test_data.csv",
        help="file that's used to compute test accuracy",
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        default="test_data.csv",
        help="new file that needs classification",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="",
        help="path to input files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="path to output files",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        default=False,
        help="whether to predict using all output, True if yes",
    )
    parser.add_argument(
        "--injtype",
        type=str,
        default="Head, Skull",
        help="The injury type for which the classifier is to be trained",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="best-classifier",
        help="The path to save the model or load the trained model from",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train", action="store_true", default=False, help="Train or evaluate"
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        default=False,
        help="Whether there is a csv file of injury descriptions that needs to be classified",
    )
    # training related
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulate_gradient_steps", type=int, default=1)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warm_up_proportion", type=float, default=0.1)
    parser.add_argument(
        "--wait_step",
        type=int,
        default=50,
        help="The number of training steps to perform after test result stops increasing",
    )
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gpu", type=str, default="0")

    # evaluation relation
    parser.add_argument(
        "--required_accuracy",
        type=float,
        default=80,
        help="No trained model will be save unless its accuracy achieved 80%",
    )

    args = parser.parse_args()

    if not args.all_models:
        args.prediction_result = "prediction_result_{}_".format(args.injtype) + args.prediction_file
    else:
        args.prediction_result = "prediction_result_" + args.prediction_file

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args)

    main(args)
