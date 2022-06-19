import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, Transformer, TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import AdamW, BertTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import matplotlib.pyplot as plt 
import argparse
import random
import os 
import time
import csv

from data import scirex_reader, pubmed_reader, TDMS_reader
from low_level_utils import *
import low_level_ReSel

def test_ll(data_list, model, args, file_path):
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
                with torch.no_grad():
                    output_logits, temp_labels, temp_acc, temp_preds = model(doc, idx, train_flag=False)
                    accuracy += temp_acc
                    nb_steps += 1
                    nb_examples += 1
            
                logits = output_logits.detach().cpu().numpy()
                labels = temp_labels
                pred = temp_preds
                # for l in label:
                rank = IR_metric(logits, labels)
                mrr += 1 / rank
                if rank <= 2:
                    hit2 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 5:
                    hit5 += 1
                
                if doc.para_labels[idx] < doc.text_data_section_numbers:
                    flag2 = 'Text'
                else:
                    flag2 = 'Table'
                csv_write.writerow([doc.gt_tuples[idx], doc.parts[doc.para_labels[idx]], doc.entities[int(doc.tup_label[idx])], doc.entities[pred], doc.entities[int(doc.tup_label[idx])] == doc.entities[pred], flag2])
                
                loss += loss
                # nb_steps += 1
            # nb_examples += len(doc.tuple_embedding)
            # doc_idx += 1
    return loss, accuracy, mrr, hit2, hit3, hit5, nb_steps, nb_examples


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
                val_list = torch.load('./saved_model/val_data.pt')
                test_list = torch.load('./saved_model/test_data.pt')
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
            # if args.model == 'hifeg':
            train_dev_list = torch.load('./saved_model/new_train_dev_pubmed.pt')
            test_list = torch.load('./saved_model/new_test_pubmed.pt')
            print('new')

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

    model = low_level_ReSel.scirex_bert(args)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    criterion = nn.CrossEntropyLoss()
    max_val_acc = 0
    max_test_acc = 0
    last_val_loss = 0
    last_val_loss2 = 0
    for epoch in range(args.epochs):
        model.train()
        train_accuracy, nb_train_examples = 0, 0
        nb_train_steps = 0
        train_loss = 0
        nb_train_loss = 0
        labels_list = []
        logits_list = []
        loss = 0
        for doc in train_list:
            if args.duplicate == 'y':
                num_iter = len(doc.gt_tuples)
            else:
                num_iter = 1
            for tup_idx in range(num_iter):
                optimizer.zero_grad()
                output_logits, labels, loss, acc = model(doc, tup_idx, train_flag=True)
                # loss += 
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                nb_train_examples += 1
                nb_train_steps += 1
        print("Train Loss: {}".format(train_loss/nb_train_steps))
        # print("Train Accuracy: {}".format(train_accuracy/nb_train_examples))

        model.eval()
        if epoch % 1 == 0:
            val_accuracy = 0
            nb_val_examples = 0
            val_loss = 0
            labels_list = []
            logits_list = []
            nb_val_steps = 0
            for doc in val_list:
                if args.duplicate == 'y':
                    num_iter = len(doc.gt_tuples)
                else:
                    num_iter = 1
                for tup_idx in range(num_iter):
                    output_logits, labels, loss, temp_val_acc = model(doc, tup_idx, train_flag=True)
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                    val_accuracy += temp_val_acc
                    nb_val_examples += 1
                    val_loss += loss.item()
                    
                    nb_val_steps += 1
            print("Validation Loss: {}".format(val_loss/nb_val_steps))
            print("Validation Accuracy: {}".format(val_accuracy/nb_val_examples))

            if val_accuracy/nb_val_examples > max_test_acc:
                print("SAVED!")
                max_test_acc = val_accuracy/nb_val_examples
                torch.save(model, './saved_model/{}_{}_{}_test.pt'.format(args.dataset, args.model, args.evaluation))

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

    train_loss, train_accuracy, train_mrr, train_hit2, train_hit3, train_hit5, nb_train_steps, nb_train_examples = test_ll(train_list, model, args, "./saved_model/csv/ll_train_results.csv")
    print("Train Accuracy: {}".format(train_accuracy/nb_train_examples))
    print("Train MRR: {}, Train Hit@2: {}, Train Hit @3: {}, Train Hit @5: {}".format(train_mrr/nb_train_examples, train_hit2/nb_train_examples, train_hit3/nb_train_examples, train_hit5/nb_train_examples))
    
    # test_tp, test_tn, test_fp, test_fn = 0, 0, 0, 0
    val_loss, val_accuracy, val_mrr, val_hit2, val_hit3, val_hit5, nb_val_steps, nb_val_examples = test_ll(val_list, model, args, "./saved_model/csv/ll_val_results.csv")
    print("Val Accuracy: {}".format(val_accuracy/nb_val_examples))
    print("Val MRR: {}, Val Hit@2: {}, Val Hit @3: {}, Val Hit @5: {}".format(val_mrr/nb_val_examples, val_hit2/nb_val_examples, val_hit3/nb_val_examples, val_hit5/nb_val_examples))
    

    test_loss, test_accuracy, test_mrr, test_hit2, test_hit3, test_hit5, nb_test_steps, nb_test_examples = test_ll(test_list, model, args, "./saved_model/csv/ll_test_results.csv")

    print("Test Accuracy: {}".format(test_accuracy/nb_test_examples))
    print("Test MRR: {}, Test Hit@2: {}, Test Hit @3: {}, Test Hit @5: {}".format(test_mrr/nb_test_examples, test_hit2/nb_test_examples, test_hit3/nb_test_examples, test_hit5/nb_test_examples))
    
    if args.softedges:
        log_path = "./saved_model/results-20220206-{}-se.log".format(args.model)
    else:
        log_path = "./saved_model/results-20220206-{}.log".format(args.model)
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