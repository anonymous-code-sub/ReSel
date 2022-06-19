import os
import torch
from torch import nn
import torch.nn.functional as F
import pickle
import random
import pandas as pd 
import numpy as np
import json
import time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import argparse
from functools import reduce
import csv
import gc
from data import scirex_reader, pubmed_reader, TDMS_reader
from high_level_utils import *
from high_level_retriever import *
import overall_ReSel

def test_ll(data_list, model, args, file_path, hl_model, score_embedding):
    accuracy = 0
    mrr = 0
    hit2 = 0
    hit3 = 0
    hit5 = 0
    input_dim = score_embedding[0].shape[-1]
    loose_acc = 0
    nb_examples = 0
    loss = 0
    labels_list = []
    preds_list = []
    criterion_list = []
    nb_steps = 0
    MAX_LEN = 512
    with open(file_path, "w") as f:
        csv_write = csv.writer(f)
        csv_header = ["Answer", "Ground-Truth para", "Ground-Truth entity", "Prediction", "RightOrNot", "Table or Text"]
        csv_write.writerow(csv_header)
        doc_idx = 0
        for doc in data_list:
            
            criterion_list.append(doc.text_data_section_numbers)
            if args.duplicate == 'y':
                num_iter = len(doc.gt_tuples)
            else:
                num_iter = 1
            for idx in range(num_iter):
                label = []
                paras = doc.para_embeddings
                paras_embed = torch.zeros((len(paras), input_dim))
                for i in range(len(paras)):
                    paras_embed[i] = torch.tensor(score_embedding[doc_idx][idx, i, :])
                    if doc.gt_tuples[idx][-1] in doc.parts[i]:
                        label.append(i)
                with torch.no_grad():
                    logits, loss = hl_model(paras_embed.to(args.device), label)
                logits = logits.detach().cpu().numpy()
                logits = logits[:,1]
                pred = np.argmax(logits)
                if pred in label:
                    with torch.no_grad():
                        output_logits, temp_labels, temp_acc, temp_preds = model(doc, idx, pred, train_flag=False)
                        accuracy += temp_acc

                    logits = output_logits.detach().cpu().numpy()
                    labels = temp_labels
                    pred = temp_preds
                    min_rank = np.inf
                    for l in labels:
                        rank = IR_metric(logits, l)
                        if rank < min_rank:
                            min_rank = rank
                    mrr += 1 / min_rank
                    if min_rank <= 2:
                        hit2 += 1
                    if min_rank <= 3:
                        hit3 += 1
                    if min_rank <= 5:
                        hit5 += 1

                loss += loss
                nb_steps += 1
                nb_examples += 1
                # nb_steps += 1
            # nb_examples += len(doc.tuple_embedding)
            doc_idx += 1
    return loss, accuracy, mrr, hit2, hit3, hit5, nb_steps, nb_examples

def initial_sim(doc_list, file_path, results_path):
    # cos = F.cosine_similarity()
    acc = 0
    mrr = 0
    hit2 = 0
    hit3 = 0
    hit5 = 0
    nb_examples = 0
    cosine_entity_scores = [] #np.zeros((len(doc_list), len(doc.tuple_embedding), len(doc.para_embeddings), len(doc.elements_embedding)))
    cosine_token_scores = [] #np.zeros((len(doc_list), len(doc.tuple_embedding), len(doc.para_embeddings), len(doc.elements_embedding)))
    string_sim_scores = [] #np.zeros((len(doc_list), len(doc.tuple_embedding), len(doc.para_embeddings), len(doc.elements_embedding)))
    lcs_scores = [] #np.zeros((len(doc_list), len(doc.tuple_embedding), len(doc.para_embeddings), len(doc.elements_embedding)))
    lcs_scores2 = []
    final_scores = []
    doc_idx = 0
    results_idx = 0
    trained_embed_scores = np.load(results_path, allow_pickle=True)
    for doc in doc_list:
        se1 = np.zeros((len(doc.tuple_embedding), len(doc.para_embeddings), len(doc.elements_embedding[0])))
        for idx in range(len(doc.tuple_embedding)):
            elements_embedding = doc.elements_embedding[idx]
            flag = True
            for elements_idx in range(len(elements_embedding)):
                for i in range(len(doc.para_embeddings)):
                    max_score = -np.inf
                    max_entity = ''
                    for j in range(len(doc.entity_ids_label[i])):
                        temp = F.cosine_similarity(elements_embedding[elements_idx].unsqueeze(0), doc.initial_embed[doc.entity_ids_label[i][j]].unsqueeze(0), dim=-1)
                        temp = temp[0]
                        if temp > max_score:
                            max_score = temp
                            max_entity = doc.parts_bert[i][j]
                    se1[idx, i, elements_idx] = max_score
            # selected_para = torch.argmax(score)
        nb_examples += len(doc.tuple_embedding)
        cosine_entity_scores.append(se1)
        # doc_idx += 1

        # doc_idx = 0
        se2 = np.zeros((len(doc.tuple_embedding), len(doc.para_embeddings), len(doc.elements_embedding[0])))
        for idx in range(len(doc.tuple_embedding)):
            elements_embedding = doc.elements_embedding[idx]
            flag = True
            for elements_idx in range(len(elements_embedding)):
                for i in range(len(doc.para_entity_embeddings)):
                    max_score = -np.inf
                    max_entity = ''
                    for j in range(len(doc.para_entity_embeddings[i])):
                        temp = F.cosine_similarity(elements_embedding[elements_idx].unsqueeze(0), doc.para_entity_embeddings[i][j].unsqueeze(0), dim=-1)
                        temp = temp[0]
                        if temp > max_score:
                            max_score = temp
                    se2[idx, i, elements_idx] = max_score
            # selected_para = torch.argmax(score)
        nb_examples += len(doc.tuple_embedding)
        cosine_token_scores.append(se2)
        # doc_idx += 1

        # doc
        se3 = np.zeros((len(doc.tuple_embedding), len(doc.para_embeddings), len(doc.elements_embedding[0])))
        se4 = np.zeros((len(doc.tuple_embedding), len(doc.para_embeddings), len(doc.elements_embedding[0])))
        se5 = np.zeros((len(doc.tuple_embedding), len(doc.para_embeddings), len(doc.elements_embedding[0])))
        for idx in range(len(doc.tuple_embedding)):
            elements_embedding = doc.elements_embedding[idx]
            flag = True
            for elements_idx in range(len(elements_embedding)):
                for i in range(len(doc.parts)):
                    max_score = -np.inf
                    max_entity = ''
                    max_score2 = -np.inf
                    max_entity2 = ''
                    max_score3 = -np.inf
                    max_entity3 = ''
                    words = doc.parts[i].split(' ')
                    for j in range(len(words)):
                        temp = LevenDist(doc.gt_tuples[idx][elements_idx], words[j])
                        temp2 = LCSSim(doc.gt_tuples[idx][elements_idx], words[j])
                        temp3 = LCSSim2(doc.gt_tuples[idx][elements_idx], words[j])
                        if 1 - temp > max_score:
                            max_score = 1 - temp
                        if temp2 > max_score2:
                            max_score2 = temp2
                        if temp3 > max_score3:
                            max_score3 = temp3
                    se3[idx, i, elements_idx] = max_score
                    se4[idx, i, elements_idx] = max_score2
                    se5[idx, i, elements_idx] = max_score3
            # selected_para = torch.argmax(score)
        nb_examples += len(doc.tuple_embedding)
        string_sim_scores.append(se3)
        lcs_scores.append(se4)
        lcs_scores2.append(se5)

        se6 = np.zeros((len(doc.tuple_embedding), len(doc.para_embeddings), 1))
        se7 = np.zeros((len(doc.tuple_embedding), len(doc.para_embeddings), 1))
        for idx in range(len(doc.tuple_embedding)):
            elements_embedding = doc.elements_embedding[idx]
            flag = True
            for i in range(len(doc.parts)):
                se6[idx, i, 0] = F.cosine_similarity(doc.tuple_embedding[idx].unsqueeze(0), doc.para_embeddings[i].unsqueeze(0), dim=-1)
                se7[idx, i, 0] = trained_embed_scores[results_idx][i]
            results_idx += 1
        doc_idx += 1
        se_final = np.concatenate((se1, se2, se3, se4, se5, se6, se7),-1)
        final_scores.append(se_final)
    return final_scores

class deep_sim(nn.Module):
    def __init__(self, input_size, hidden_size, num_aspects):
        super(deep_sim, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_aspects = num_aspects
        self.fc1 = nn.Sequential(nn.Linear(self.num_aspects*2, self.hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size, 2))
        self.fc2 = nn.Sequential(nn.Linear(self.num_aspects*3, self.hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size, 2))
        self.fc3 = nn.Sequential(nn.Linear(2, self.hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size, 2))
        self.fc = nn.Sequential(nn.Linear(self.num_aspects*5+2, self.hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size, 2))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, para_embeds, label):

        logits = self.softmax(self.fc(para_embeds))

        loss = torch.sum(-torch.log(logits[:, 0])) 
        for l in label:
            loss += torch.log(logits[l, 0]) - torch.log(logits[l, 1])

        return logits, loss

def test_ds(data_list, model, args, file_path, score_embedding, results):
    accuracy = 0
    mrr = 0
    hit2 = 0
    hit3 = 0
    hit5 = 0
    loose_acc = 0
    nb_examples = 0
    loss = 0
    labels_list = []
    preds_list = []
    criterion_list = []
    high_level_answers = []
    nb_steps = 0
    MAX_LEN = 512
    input_dim = score_embedding[0].shape[-1]
    with open(file_path, "a+") as f:
        csv_write = csv.writer(f)
        csv_header = ["Answer", "Ground-Truth", "Prediction", "RightOrNot", "Another Answer?", "Table or Text"]
        csv_write.writerow(csv_header)
        doc_idx = 0
        for doc in data_list:
            answer = []
            for idx in range(len(doc.tuple_embedding)):
                criterion_list.append(doc.text_data_section_numbers)
                query = doc.tuple_embedding[idx]
                # label = doc.para_labels[idx]
                label = []
                paras = doc.para_embeddings
                paras_embed = torch.zeros((len(paras), input_dim))
                labels = torch.zeros(len(paras))
                labels[label] = 1
                for i in range(len(paras)):
                    paras_embed[i] = torch.tensor(score_embedding[doc_idx][idx, i, :])
                    if doc.gt_tuples[idx][-1] in doc.parts[i]:
                        label.append(i)
                with torch.no_grad():
                    logits, loss = model(paras_embed.to(args.device), label)
                logits = logits.detach().cpu().numpy()
                labels = labels.numpy()
                min_rank = np.inf
                for l in label:
                    rank = IR_metric(logits[:, 1], l)
                    if rank < min_rank:
                        min_rank = rank
                mrr += 1 / min_rank
                if min_rank <= 2:
                    hit2 += 1
                if min_rank <= 3:
                    hit3 += 1
                if min_rank <= 5:
                    hit5 += 1
                label = doc.para_labels[idx]
                pred = np.argmax(logits[:, 1])
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                # output_logits = output_logits.detach().to('cpu').numpy()
                # labels = labels.to('cpu').numpy()
                labels_list.append(label)
                preds_list.append(pred)
                answer.append(logits[:,1])
                if label == pred:
                    accuracy += 1
                # if query[-1] in doc.parts[pred]:
                #     loose_acc += 1
                # val_accuracy += flat_accuracy(logits, labels)
                if doc.gt_tuples[idx][-1] in doc.parts[pred]:
                    flag = True
                    loose_acc += 1
                else:
                    flag = False
                if label < doc.text_data_section_numbers:
                    flag2 = 'Text'
                else:
                    flag2 = 'Table'
                csv_write.writerow([doc.gt_tuples[idx], doc.parts[label], doc.parts[pred], label == pred, flag, flag2])
                
                loss += loss.item()
                nb_steps += 1
            nb_examples += len(doc.tuple_embedding)
            doc_idx += 1
            high_level_answers.append(answer)
    
    np.save(results, np.array(high_level_answers))
    return loss, accuracy, mrr, hit2, hit3, hit5, loose_acc, nb_steps, nb_examples, labels_list, preds_list, criterion_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.002, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    # parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-9, type=float, help="Parameter Epsilon for AdamW optimizer.")
    parser.add_argument("--epochs", default=10, type=int, help="The number of the epochs.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="The max value of the gradient normalization.")
    parser.add_argument("--model", default='bert-base', type=str, help="The pretrained model.")
    parser.add_argument("--use_entity_type", action="store_true", help="Whether use entity_type or not.")
    parser.add_argument("--use_entity_ids", action="store_true", help="Whether use entity_ids or not.")
    parser.add_argument("--partial", action="store_true", help="Use parts of the documents as candidate pools.")
    parser.add_argument("--entity_type_embed_size", default=768, type=int, help="The size of the entity type embedding.")
    parser.add_argument("--saved_embed", default='n', type=str, help="Use saved embedding or not.")
    parser.add_argument("--saved_embed2", default='n', type=str, help="Use saved embedding or not.")
    parser.add_argument("--gcn_layers", default=1, type=int, help="The number of the GCN layers.")
    parser.add_argument("--bert_model", default='base', type=str, help="The type of bert model.")
    parser.add_argument("--embed_style", default='sentences', type=str, help="The pretrained model.")
    # parser.add_argument("--partial", action="store_true", help="Use parts of the documents as candidate pools.")
    parser.add_argument("--gamma", default=0.99, type=float, help="The discount factor.")
    parser.add_argument("--max_episode", default=5000, type=int, help="The max number of the espisodes.")
    parser.add_argument("--focus", default='score', type=str, help="The query element in the tuple.")
    parser.add_argument("--query_style", default='tuple', type=str, help="The query embedding style [tuple/question].")
    parser.add_argument("--bfs", action="store_true", help="Apply BFS to obtain the ground-truth paths or not.")
    parser.add_argument("--table_style", default='table', type=str, help="The table embedding style [table/caption/caption+table].")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset or not.")
    parser.add_argument("--duplicate", default='y', type=str, help="Allow duplication of tuples in a single document or not.")
    parser.add_argument("--initialize", action="store_true", help="Initialize with pre-trained weights.")
    parser.add_argument("--softedges", action="store_true", help="Create similarity-based soft edges or not.")
    parser.add_argument("--evaluation", default="multiplication", help="Multiplication score or classification score.")
    parser.add_argument("--edges", default="ccr", help="Use what kinds of edges in the graph model.")
    parser.add_argument("--gat_headers", default=1, type=int, help="The number of the GAT headers.")
    parser.add_argument("--dataset", default='scirex', type=str, help="The dataset: scirex/pubmed")
    parser.add_argument("--lepochs", default=10, type=int, help="The number of the epochs.")
    parser.add_argument("--llr", default=0.002, type=float, help="The initial learning rate for Adam.")
    

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    if args.dataset == 'scirex':
        if args.saved_embed == 'n':
            time_start = time.time()
            train_path = './datasets/scirex/train.jsonl'
            train_list = scirex_reader.read_files(args, train_path)
            time_end = time.time()
            print('Time cost of processing Training Set: ',time_end - time_start,'s')

            time_start = time.time()
            val_path = './datasets/scirex/dev.jsonl'
            val_list = scirex_reader.read_files(args, val_path)
            time_end = time.time()
            print('Time cost of processing Validation Set: ',time_end - time_start,'s')

            time_start = time.time()
            test_path = './datasets/scirex/test.jsonl'
            test_list = scirex_reader.read_files(args, test_path)
            time_end = time.time()
            print('Time cost of processing Testing Set: ',time_end - time_start,'s')
            if args.softedges:
                torch.save(train_list, './saved_model/train_se_data.pt')
                torch.save(val_list, './saved_model/val_se_data.pt')
                torch.save(test_list, './saved_model/test_se_data.pt')
            else:
                torch.save(train_list, './saved_model/train_data.pt')
                torch.save(val_list, './saved_model/val_data.pt')
                torch.save(test_list, './saved_model/test_data.pt')
        else:
            if args.softedges:
                train_list = torch.load('./saved_model/train_new_data.pt')
                val_list = torch.load('./saved_model/val_new_data.pt')
                test_list = torch.load('./saved_model/test_new_data.pt')
            else:
                train_list = torch.load('./saved_model/train_data.pt')
                val_list = torch.load('./mol-sci/saved_model/val_data.pt')
                test_list = torch.load('./mol-sci/saved_model/test_data.pt')
        if args.shuffle:
            data_list = train_list + val_list + test_list
            len_data = len(data_list)
            random.shuffle(data_list)
            train_list = data_list[:int(0.6*len_data)]
            val_list = data_list[int(0.6*len_data):int(0.8*len_data)]
            test_list = data_list[int(0.8*len_data):]
        num_aspects = 4
    elif args.dataset == 'pubmed':
        if args.saved_embed == 'n':
            time_start = time.time()
            train_dev_path = './datasets/pubmed/data/examples/document/ds_train_dev.txt'
            train_dev_list = pubmed_reader.read_files(args, train_dev_path)
            time_end = time.time()
            print('Time cost: ',time_end - time_start,'s')

            time_start = time.time()
            test_path = './datasets/pubmed/data/examples/document/jax_dev_test.txt'
            test_list = pubmed_reader.read_files(args, test_path)
            time_end = time.time()
            print('Time cost: ',time_end - time_start,'s')

            torch.save(train_dev_list, './saved_model/new_train_dev_pubmed.pt')
            torch.save(test_list, './saved_model/new_test_pubmed.pt')
        else:
            train_dev_list = torch.load('./saved_model/new_train_dev_pubmed.pt')
            test_list = torch.load('./saved_model/new_test_pubmed.pt')
        if args.shuffle:
            len_data = len(train_dev_list)
            random.shuffle(train_dev_list)
            train_list = train_dev_list[:int(0.8*len_data)]
            val_list = train_dev_list[int(0.8*len_data):]
        num_aspects = 2
    elif args.dataset == 'tdms':
        if args.saved_embed == 'n':
            time_start = time.time()
            pdf_txt_path = './datasets/TDMS/dataset/pdfFile_txt'
            pdf_table_path = './datasets/TDMS/dataset/pdfFile_table'
            tdms_tuple_path = './datasets/TDMS/dataset/resultsAnnotation.tsv'
            train_dev_test_list = TDMS_reader.read_files(args, pdf_txt_path, pdf_table_path, tdms_tuple_path)
            time_end = time.time()
            print('Time cost: ',time_end - time_start,'s')

            torch.save(train_dev_test_list, './saved_model/train_dev_test_tdms.pt')
        else:
            train_dev_test_list = torch.load('./saved_model/train_dev_test_tdms.pt')
        if args.shuffle:
            len_data = len(train_dev_test_list)
            random.shuffle(train_dev_list)
            train_list = train_dev_test_list[:int(0.6*len_data)]
            val_list = train_dev_test_list[int(0.6*len_data):int(0.8*len_data)]
            test_list = train_dev_test_list[int(0.8*len_data):]
        num_aspects = 3


    hidden_state_embed = 768

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # model.to(args.device)

    max_num = 0
    for doc in train_list:
        if len(doc.para_embeddings) > max_num:
            max_num = len(doc.para_embeddings)
    print("The number of possible choices in Training set: {}".format(max_num))
    
    max_num = 0
    for doc in val_list:
        if len(doc.para_embeddings) > max_num:
            max_num = len(doc.para_embeddings)
    print("The number of possible choices in Validation set: {}".format(max_num))

    max_num = 0
    for doc in test_list:
        if len(doc.para_embeddings) > max_num:
            max_num = len(doc.para_embeddings)  
    print("The number of possible choices in Test set: {}".format(max_num))

    if args.saved_embed2 == 'n':
        train_score_embedding = initial_sim(train_list, "./saved_model/csv/eps_train_results.csv", "./saved_model/csv/para_selector_train_results_hl_{}.npy".format(args.seed))
        val_score_embedding = initial_sim(val_list, "./saved_model/csv/eps_val_results.csv", "./saved_model/csv/para_selector_val_results_hl_{}.npy".format(args.seed))
        test_score_embedding = initial_sim(test_list, "./saved_model/csv/eps_test_results.csv", "./saved_model/csv/para_selector_test_results_hl_{}.npy".format(args.seed))

        torch.save(train_score_embedding, './saved_model/saved_embed/{}_train_score_embedding_22_{}.pt'.format(args.dataset, args.seed))
        torch.save(val_score_embedding, './saved_model/saved_embed/{}_val_score_embedding_22_{}.pt'.format(args.dataset, args.seed))
        torch.save(test_score_embedding, './saved_model/saved_embed/{}_test_score_embedding_22_{}.pt'.format(args.dataset, args.seed))
    else:
        train_score_embedding = torch.load('./saved_model/saved_embed/{}_train_score_embedding_22_{}.pt'.format(args.dataset, args.seed))
        val_score_embedding = torch.load('./saved_model/saved_embed/{}_val_score_embedding_22_{}.pt'.format(args.dataset, args.seed))
        test_score_embedding = torch.load('./saved_model/saved_embed/{}_test_score_embedding_22_{}.pt'.format(args.dataset, args.seed))

    input_dim = train_score_embedding[0].shape[-1]
    hidden_dim = 64
    hl_model = torch.load('./saved_model/hl_mv_' + args.dataset + '_test.pt')
    hl_model.eval()
        
    ll_model = overall_ReSel.scirex_bert(args)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(ll_model)
    ll_model.to(args.device)

    optimizer = torch.optim.Adam(ll_model.parameters(), lr=args.llr, eps=args.adam_epsilon)
    criterion = nn.CrossEntropyLoss()
    max_val_acc = 0
    max_test_acc = 0
    last_val_loss = 0
    last_val_loss2 = 0
    for epoch in range(args.lepochs):
        ll_model.train()
        train_accuracy, nb_train_examples = 0, 0
        nb_train_steps = 0
        train_loss = 0
        nb_train_loss = 0
        labels_list = []
        logits_list = []
        loss = 0
        doc_id = 0
        for doc in train_list:

            if args.duplicate == 'y':
                num_iter = len(doc.gt_tuples)
            else:
                num_iter = 1

            for tup_idx in range(num_iter):
                label = []
                paras = doc.para_embeddings
                paras_embed = torch.zeros((len(paras), input_dim))
                for i in range(len(paras)):
                    paras_embed[i] = torch.tensor(train_score_embedding[doc_id][tup_idx, i, :])
                    if doc.gt_tuples[tup_idx][-1] in doc.parts[i]:
                        label.append(i)
                with torch.no_grad():
                    logits, loss = hl_model(paras_embed.to(args.device), label)
                logits = logits.detach().cpu().numpy()
                logits = logits[:,1]
                pred = np.argmax(logits)
                # if doc.para_labels[tup_idx] in logits.argsort()[-2:]:
                optimizer.zero_grad()
                output_logits, labels, loss, acc = ll_model(doc, tup_idx, doc.para_labels[tup_idx], train_flag=True)
                # loss += 
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                nb_train_examples += 1
                nb_train_steps += 1
            doc_id += 1
        print("Train Loss: {}".format(train_loss/nb_train_steps))
        # print("Train Accuracy: {}".format(train_accuracy/nb_train_examples))

        ll_model.eval()
        if epoch % 1 == 0:
            val_accuracy = 0
            nb_val_examples = 0
            val_loss = 0
            labels_list = []
            logits_list = []
            nb_val_steps = 0
            doc_id = 0
            for doc in val_list:
                if args.duplicate == 'y':
                    num_iter = len(doc.gt_tuples)
                else:
                    num_iter = 1
                for tup_idx in range(num_iter):
                    label = []
                    paras = doc.para_embeddings
                    paras_embed = torch.zeros((len(paras), input_dim))
                    for i in range(len(paras)):
                        paras_embed[i] = torch.tensor(val_score_embedding[doc_id][tup_idx, i, :])
                        if doc.gt_tuples[tup_idx][-1] in doc.parts[i]:
                            label.append(i)
                    with torch.no_grad():
                        logits, loss = hl_model(paras_embed.to(args.device), label)
                    logits = logits.detach().cpu().numpy()
                    logits = logits[:,1]
                    pred = np.argmax(logits)
                    if pred in label:
                        output_logits, labels, loss, temp_val_acc = ll_model(doc, tup_idx, pred, train_flag=True)
                        if torch.cuda.device_count() > 1:
                            loss = loss.mean()

                        val_accuracy += temp_val_acc
                        val_loss += loss.item()
                    nb_val_examples += 1
                    
                    
                    nb_val_steps += 1
                doc_id += 1
            print("Validation Loss: {}".format(val_loss/nb_val_steps))
            print("Validation Accuracy: {}".format(val_accuracy/nb_val_examples))
            # val_precision = val_tp/(val_tp+val_fp+1e-5)
            test_accuracy = 0
            nb_test_examples = 0
            test_loss = 0
            labels_list = []
            logits_list = []
            nb_test_steps = 0
            # max_val_acc = -1
            test_tp, test_tn, test_fp, test_fn = 0, 0, 0, 0
            doc_id = 0
            for doc in test_list:
                if args.duplicate == 'y':
                    num_iter = len(doc.gt_tuples)
                else:
                    num_iter = 1
                for tup_idx in range(num_iter):
                    label = []
                    paras = doc.para_embeddings
                    paras_embed = torch.zeros((len(paras), input_dim))
                    for i in range(len(paras)):
                        paras_embed[i] = torch.tensor(test_score_embedding[doc_id][tup_idx, i, :])
                        if doc.gt_tuples[tup_idx][-1] in doc.parts[i]:
                            label.append(i)
                    with torch.no_grad():
                        logits, loss = hl_model(paras_embed.to(args.device), label)
                    logits = logits.detach().cpu().numpy()
                    logits = logits[:,1]
                    pred = np.argmax(logits)
                    if pred in label:
                        output_logits, labels, loss, temp_test_acc = ll_model(doc, tup_idx, pred, train_flag=True)
                        if torch.cuda.device_count() > 1:
                            loss = loss.mean()
                        test_accuracy += temp_test_acc
                    nb_test_examples += 1
                
                doc_id += 1
                test_loss += loss.item()
                
                nb_test_steps += 1
            print("Test Loss: {}".format(test_loss/nb_test_steps))
            print("Test Accuracy: {}".format(test_accuracy/nb_test_examples))
            if val_accuracy/nb_val_examples > max_test_acc:
                print("SAVED!")
                max_test_acc = val_accuracy/nb_val_examples
                torch.save(ll_model, './saved_model/{}_{}_{}_test.pt'.format(args.dataset, args.model, args.evaluation))

            # if val_loss/nb_val_steps > last_val_loss and last_val_loss > last_val_loss2: # early stopping
            #     break
            last_val_loss2 = last_val_loss
            last_val_loss = val_loss / nb_val_steps
    
    test_accuracy = 0
    nb_test_examples = 0
    nb_test_steps = 0
    test_tp, test_tn, test_fp, test_fn = 0, 0, 0, 0
    model = torch.load('./saved_model/{}_{}_{}_test.pt'.format(args.dataset, args.model, args.evaluation))
    model.eval()
    val_accuracy = 0
    nb_val_examples = 0
    nb_val_steps = 0

    train_loss, train_accuracy, train_mrr, train_hit2, train_hit3, train_hit5, nb_train_steps, nb_train_examples = test_ll(train_list, model, args, "/localscratch/yzhuang43/mol-sci/saved_model/csv/overall_train_results.csv", hl_model, train_score_embedding)
    print("Train Accuracy: {}".format(train_accuracy/nb_train_examples))
    print("Train MRR: {}, Train Hit@2: {}, Train Hit @3: {}, Train Hit @5: {}".format(train_mrr/nb_train_examples, train_hit2/nb_train_examples, train_hit3/nb_train_examples, train_hit5/nb_train_examples))
    
    # test_tp, test_tn, test_fp, test_fn = 0, 0, 0, 0
    val_loss, val_accuracy, val_mrr, val_hit2, val_hit3, val_hit5, nb_val_steps, nb_val_examples = test_ll(val_list, model, args, "/localscratch/yzhuang43/mol-sci/saved_model/csv/overall_val_results.csv", hl_model, val_score_embedding)
    print("Val Accuracy: {}".format(val_accuracy/nb_val_examples))
    print("Val MRR: {}, Val Hit@2: {}, Val Hit @3: {}, Val Hit @5: {}".format(val_mrr/nb_val_examples, val_hit2/nb_val_examples, val_hit3/nb_val_examples, val_hit5/nb_val_examples))
    

    test_loss, test_accuracy, test_mrr, test_hit2, test_hit3, test_hit5, nb_test_steps, nb_test_examples = test_ll(test_list, model, args, "/localscratch/yzhuang43/mol-sci/saved_model/csv/overall_test_results.csv", hl_model, test_score_embedding)

    print("Test Accuracy: {}".format(test_accuracy/nb_test_examples))
    print("Test MRR: {}, Test Hit@2: {}, Test Hit @3: {}, Test Hit @5: {}".format(test_mrr/nb_test_examples, test_hit2/nb_test_examples, test_hit3/nb_test_examples, test_hit5/nb_test_examples))
    
    if args.softedges:
        log_path = "/localscratch/yzhuang43/mol-sci/saved_model/results-20220206-{}-se.log".format(args.model)
    else:
        log_path = "/localscratch/yzhuang43/mol-sci/saved_model/results-20220206-{}.log".format(args.model)
    with open(log_path, "a") as f:
        f.write(str(time.time()))
        f.write('\n')
        f.write(args.model+' '+args.evaluation+' '+args.edges+str(args.gcn_layers))
        f.write('\n')
        f.write("Train Accuracy: {}".format(train_accuracy/nb_train_examples))
        f.write('\n')
        f.write("Train MRR: {}, Train Hit@2: {}, Train Hit @3: {}, Train Hit @5: {}".format(train_mrr/nb_train_examples, train_hit2/nb_train_examples, train_hit3/nb_train_examples, train_hit5/nb_train_examples))
        f.write('\n')
        f.write("Val Accuracy: {}".format(val_accuracy/nb_val_examples))
        f.write('\n')
        f.write("Val MRR: {}, Val Hit@2: {}, Val Hit @3: {}, Val Hit @5: {}".format(val_mrr/nb_val_examples, val_hit2/nb_val_examples, val_hit3/nb_val_examples, val_hit5/nb_val_examples))
        f.write('\n')
        f.write("Test Accuracy: {}".format(test_accuracy/nb_test_examples))
        f.write('\n')
        f.write("Test MRR: {}, Test Hit@2: {}, Test Hit @3: {}, Test Hit @5: {}".format(test_mrr/nb_test_examples, test_hit2/nb_test_examples, test_hit3/nb_test_examples, test_hit5/nb_test_examples))
        f.write('\n')
        f.write('---------------------------------------\n')


if __name__ == "__main__":
    main()

