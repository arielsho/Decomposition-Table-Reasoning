from transformers import TapasConfig, TapasForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import TapasTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import pandas as pd
import pdb
import os
import torch
from torch.nn import CrossEntropyLoss
import random
import time
import argparse
from tqdm import tqdm
import numpy as np
from models import DoubleBERT
from tensorboardX import SummaryWriter


def get_evidence_features(evidence, tokenizer_bert):
    evidence_lst = evidence.split('. ')
    evd_num = len(evidence_lst)
    if evd_num < 2:
        pad_evd(evidence_lst)
    else:
        evidence_lst = evidence_lst[:2]
        evd_num = len(evidence_lst)

    tokens_tmp = []
    for evd in evidence_lst:
        if not evd.endswith('.'):
            evd += '.'
        evd_tmp = tokenizer_bert.tokenize(evd)
        tokens_tmp.append(evd_tmp)
    _truncate_seq_pair_verb(tokens_tmp, args.max_seq_length_evidence - 4)

    tokens = []
    segment_ids = []
    cls_ids = []
    cls_mask = [0, 0]
    for i in range(evd_num):
        cls_mask[i] = 1

    for idx, item in enumerate(tokens_tmp):
        cls_ids.append(len(tokens))
        tokens.extend(["[CLS]"] + item + ["[SEP]"])
        segment_ids += [idx % 2] * (len(item) + 2)
        assert len(tokens) == len(segment_ids)

    input_ids = tokenizer_bert.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (args.max_seq_length_evidence - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == args.max_seq_length_evidence
    assert len(input_mask) == args.max_seq_length_evidence
    assert len(segment_ids) == args.max_seq_length_evidence
    assert len(cls_ids) == len(cls_mask)
    assert cls_ids[0] < 64 and cls_ids[1] < 63

    return input_ids, input_mask, segment_ids, cls_ids, cls_mask


def pad_evd(evd_lst):
    while len(evd_lst) < 2:
        evd_lst.append("[PAD]")


def _truncate_seq_pair_verb(tokens_tmp, max_length):
    while True:
        total_length = 0
        max_idx = 0
        max_idx_next = 0
        max_len = 0
        for i, tokens in enumerate(tokens_tmp):
            total_length += len(tokens)
            if len(tokens) > max_len:
                max_idx_next = max_idx
                max_len = len(tokens)
                max_idx = i
        if total_length <= max_length:
            break
        if len(tokens_tmp[max_idx]) > 2:
            tokens_tmp[max_idx].pop()
        else:
            tokens_tmp[max_idx_next].pop()


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, tokenizer_bert):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer_bert = tokenizer_bert

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        table = pd.read_csv(args.data_dir + '/' + item.table_file).astype(str)
        encoding = self.tokenizer(table=table, queries=item.question, truncation=True,
                                  padding="max_length", return_tensors="pt")

        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = int(item.answer_text)
        assert encoding['labels'] in [0, 1]
        encoding['labels'] = torch.tensor(encoding['labels'], dtype=torch.long)

        evidence = item.evidence
        input_ids, input_mask, segment_ids, cls_ids, cls_mask = get_evidence_features(evidence, tokenizer_bert=self.tokenizer_bert)
        encoding["input_ids_evd"] = torch.tensor(input_ids, dtype=torch.long)
        encoding["attention_mask_evd"] = torch.tensor(input_mask, dtype=torch.long)
        encoding["token_type_ids_evd"] = torch.tensor(segment_ids, dtype=torch.long)
        encoding["cls_ids_evd"] = torch.tensor(cls_ids, dtype=torch.long)
        encoding["cls_mask_evd"] = torch.tensor(cls_mask, dtype=torch.long)

        return encoding

    def __len__(self):
        return len(self.data)


def get_dataloader(data_dir, file_name, batch_size, tokenizer, tokenizer_bert, phase):

    data = pd.read_csv(os.path.join(data_dir, file_name), sep='\t')
    dataset = TableDataset(data, tokenizer, tokenizer_bert)
    if phase == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


def _mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print(">> mkdir: {}".format(path))


def run_train(device, tokenizer, tokenizer_bert, model, writer, phase="train"):

    model.train()

    num_train_optimization_steps = int(92283 / args.batch_size / args.gradient_accumulation_steps) * args.epoch
    warmup_steps = int(args.warmup_rate * num_train_optimization_steps)

    param_optimizer = []
    if args.tune_tapas_after10k:
        for item in list(model.named_parameters()):
            if "Tapas" not in item[0]:
                param_optimizer.append(item[1])
    else:
        param_optimizer = list(model.parameters())
    optimizer = AdamW(param_optimizer, lr=args.lr_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    optimizer.zero_grad()

    train_dataloader = get_dataloader(data_dir=args.data_dir, file_name=args.train_file, batch_size=args.batch_size,
                                      tokenizer=tokenizer, tokenizer_bert=tokenizer_bert, phase=phase)

    global_step = 0
    best_acc = 0.0

    loss_fct = CrossEntropyLoss()
    optimizer_flag = 0

    # training
    for epoch in range(args.epoch):
        for idx, batch in enumerate(tqdm(train_dataloader)):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            input_ids_evd = batch["input_ids_evd"].to(device)
            attention_mask_evd = batch["attention_mask_evd"].to(device)
            token_type_ids_evd = batch["token_type_ids_evd"].to(device)

            cls_ids_evd = batch["cls_ids_evd"].to(device)
            cls_mask_evd = batch["cls_mask_evd"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels, input_ids_evd=input_ids_evd, input_mask_evd=attention_mask_evd,
                            segment_ids_evd=token_type_ids_evd, cls_ids_evd=cls_ids_evd, cls_mask_evd=cls_mask_evd)

            loss = loss_fct(logits.view(-1, 2), labels.view(-1))


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if args.tune_tapas_after10k:
                if global_step > args.optimizer_switch_step and optimizer_flag == 0:
                    optimizer_flag = 1
                    print(">> start to tune BERT-evidence after {}k global steps...".format(args.optimizer_switch_step / 1000))
                    param_optimizer = list(model.parameters())
                    optimizer = AdamW(param_optimizer, lr=args.lr_rate)
                    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                num_training_steps=num_train_optimization_steps)

            # do eval
            if (idx + 1) % (args.period * args.gradient_accumulation_steps) == 0:
                model.eval()
                model_to_save = model.module if hasattr(model, 'module') else model

                eval_phase = "dev"
                if args.do_complex_test:
                    eval_phase = "complex_test"

                dev_acc = run_eval(device, tokenizer, tokenizer_bert, model, global_step, writer=writer, phase=eval_phase)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    torch.save(model_to_save.state_dict(), '{}/epoch_{}_{:.3}k_step_{:.4}.pt'.format(args.save_model, epoch, global_step/1000, dev_acc*100))
                    print("**** Now the best acc is {:.4}, epoch: {}****\n".format(dev_acc, epoch))
                model.train()
        torch.save(model.state_dict(), '{}/epoch_{}.pt'.format(args.save_model, epoch))


def run_eval(device, tokenizer, tokenizer_bert, model, global_step=-1, writer=None, phase=None):
    model.eval()

    data_file = {"dev": "dev.tsv",
                 "std_test": "test.tsv",
                 "complex_test": "complex_test.tsv",
                 "simple_test": "simple_test.tsv",
                 "small_test": "small_test.tsv"}

    eval_dataloader = get_dataloader(data_dir=args.data_dir, file_name=data_file[phase], batch_size=args.eval_batch_size,
                                     tokenizer=tokenizer, tokenizer_bert=tokenizer_bert, phase=phase)

    preds = []
    all_labels = []
    eval_loss = 0.0
    num_steps = 0

    loss_fct = CrossEntropyLoss()

    for idx, batch in enumerate(tqdm(eval_dataloader, desc=phase)):
        num_steps += 1

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        input_ids_evd = batch["input_ids_evd"].to(device)
        attention_mask_evd = batch["attention_mask_evd"].to(device)
        token_type_ids_evd = batch["token_type_ids_evd"].to(device)

        cls_ids_evd = batch["cls_ids_evd"].to(device)
        cls_mask_evd = batch["cls_mask_evd"].to(device)

        gold_answers = batch['labels']

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                       labels=labels, input_ids_evd=input_ids_evd, input_mask_evd=attention_mask_evd,
                       segment_ids_evd=token_type_ids_evd, cls_ids_evd=cls_ids_evd, cls_mask_evd=cls_mask_evd)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        eval_loss += loss.mean().item()
        batch_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
        preds.extend(batch_pred.tolist())
        all_labels.extend(gold_answers.detach().cpu().numpy().tolist())

    eval_loss /= num_steps

    assert len(all_labels) == len(preds)
    correct = 0
    for idx, g_label in enumerate(all_labels):
        p_label = preds[idx]
        if p_label == g_label:
            correct += 1
    acc = correct / len(all_labels)
    print(">> Phase: {}, Accuracy: {:.4f}, Loss: {:.5}".format(phase, acc*100, eval_loss))
    model.train()

    if global_step != -1 and writer is not None:
        writer.add_scalar('{}/{}'.format(phase, "{}_loss".format(phase)), eval_loss, global_step)
        writer.add_scalar('{}/{}'.format(phase, "{}_acc".format(phase)), acc*100, global_step)
        print(">> Global step: {}, Len of dataset: {}".format(global_step, len(all_labels)))

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_tapas_after10k", action='store_true')
    parser.add_argument("--optimizer_switch_step", default=10000, help="10000 is default")
    parser.add_argument("--data_dir", type=str, help="input data dir")
    parser.add_argument("--train_file", default="train.tsv", help="can be train_complex.tsv")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_complex_test", action="store_true")
    parser.add_argument("--do_simple_test", action="store_true")
    parser.add_argument("--do_small_test", action="store_true")

    parser.add_argument("--model", default="google/tapas-base-finetuned-tabfact")
    parser.add_argument("--load_tapas_model", type=str)
    parser.add_argument("--load_model", type=str)

    parser.add_argument("--lr_rate", default=2e-5, help="5e-5")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epoch", default=15)
    parser.add_argument("--period", default=2000)
    parser.add_argument("--gradient_accumulation_steps", default=2)
    parser.add_argument("--warmup_rate", default=0.2)

    parser.add_argument("--save_model", default="./outputs/{}".format(time.strftime("%m%d-%H%M%S")))
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    # bert_model for extra evidence
    parser.add_argument("--bert_model", type=str, default="google/tapas-base-finetuned-tabfact")
    parser.add_argument("--max_seq_length_evidence", default=64)
    parser.add_argument("--in_dim", default=768)
    parser.add_argument("--mem_dim", default=768)

    args = parser.parse_args()

    _mkdir('./outputs')
    if args.do_train:
        _mkdir(args.save_model)
    tokenizer = TapasTokenizer.from_pretrained(args.model)
    tokenizer_bert = TapasTokenizer.from_pretrained(args.bert_model)
    device = torch.device('cuda')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    writer = SummaryWriter(os.path.join(args.save_model, 'events'))

    model = DoubleBERT(args)
    if args.load_model:
        print(">> Load Model: {}".format(args.load_model))
        model.load_state_dict(torch.load(args.load_model))
    elif args.load_tapas_model:
        print(">> Load Model: {} + {}".format(args.load_tapas_model, args.bert_model))
    else:
        print(">> Load Model: {} + {}".format(args.model, args.bert_model))

    print(">> Load data: {}".format(args.data_dir))
    # print(">> random seed: {}".format(args.seed))

    # build pipeline
    if args.do_train:
        run_train(device, tokenizer, tokenizer_bert, model, writer=writer, phase="train")

    if args.do_eval:
        run_eval(device, tokenizer, tokenizer_bert, model, global_step=-1, writer=None, phase="dev")

    if args.do_test:
        run_eval(device, tokenizer, tokenizer_bert, model, global_step=-1, writer=None, phase="std_test")

    if args.do_simple_test:
        run_eval(device, tokenizer, tokenizer_bert, model, global_step=-1, writer=None, phase="simple_test")

    if args.do_complex_test:
        run_eval(device, tokenizer, tokenizer_bert, model, global_step=-1, writer=None, phase="complex_test")

    if args.do_small_test:
        run_eval(device, tokenizer, tokenizer_bert, model, global_step=-1, writer=None, phase="small_test")


    '''
    train:
        CUDA_VISIBLE_DEVICES=2 python run.py --do_train --do_eval --tune_tapas_after10k --load_tapas_model ../ckpt/base.pt --data_dir ../Data/TabFact_data
    test:
        CUDA_VISIBLE_DEVICES=2 python run.py --do_eval --do_test --do_simple_test --do_complex_test --do_small_test --tune_tapas_after10k --load_model ../ckpt/model.pt --data_dir ../Data/TabFact_data
    '''