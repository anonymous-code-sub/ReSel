from email.policy import default
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize
from matplotlib.pyplot import text
import numpy as np
import torch 
import argparse
import random
import time
from transformers import BertModel, BertTokenizer
import os
from TDMS_utils import *
from collections import OrderedDict
import tokenizations
import json
from transformers import *


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class TDMSdataloader:
    def __init__(self, filename, text_json, table_seq, table_caption, tuples, tokenizer, BERTmodel, args):
        self.id = filename
        self.device = args.device
        self.args = args
        flag = True
        for relation_tuple in tuples:
            if relation_tuple[3] != '-':
                flag = False
        if len(table_seq) != len(table_caption):
            flag = True
        if flag: 
            print('ERROR')
            return        
        # print(table_seq)
        # input()
        table_words = []
        for i in range(len(table_seq)):
            table_words += table_caption[i].split(' ')
            for j in range(len(table_seq[i])):
                table_words += table_seq[i][j]
        # Tokenize_passage 
        input_ids, tokenized_text_json = self.tokenize_passage(text_json, tokenizer, table_seq, table_caption)

        # Words + tables
        words = [word.lower() for section in tokenized_text_json for paragraph in tokenized_text_json[section] for word in paragraph]
        words = words + [t.lower() for t in table_words]
        # words += table_seq

        # Generate paragraph ids
        # words, paragraph_ids = self.find_paragraph_ids(tokenized_text_json, words, table_seq)
        
        # Record the TDMS relation tuples
        self.read_tuple_labels(args, tuples, words, tokenizer, BERTmodel)

        # Padding the paragraph and set up sentence mask
        self.BERT_encoder(args, input_ids, BERTmodel)


    def tokenize_passage(self, text_json, tokenizer, table_seq, table_caption):
        ner_tokenizer = BertTokenizer.from_pretrained('./saved_model/BERT-base-new-0-03-bio', do_lower_case=False)
        ner_model = BertForTokenClassification.from_pretrained('./saved_model/BERT-base-new-0-03-bio', num_labels=10)
        ner_model.to(self.device)
        ner_model.eval()

        entity_type_dict = {"O":0, "B-Material":1, "I-Material":2, "B-Method":3, "I-Method":4, "B-Metric":5, "I-Metric":6, "B-Task":7, "I-Task":8, "score":9}
        inverse_entity_type_dict = {0:"O", 1:"B-Material", 2:"I-Material", 3:"B-Method", 4:"I-Method", 5:"B-Metric", 6:"I-Metric", 7:"B-Task", 8:"I-Task", 9:"score"}
        
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        input_ids = []
        tokenized_json = OrderedDict()
        self.entity_ids_set = []
        self.entities = []
        self.entity_types = []
        self.entity_ids = []
        self.section_ids = []
        self.parts = []
        section_id = 0
        for section in text_json:
            for paragraph in text_json[section]:
                
                tokenized_paragraph = ['[CLS]'] + tokenizer.tokenize(paragraph) + ['[SEP]']
                self.parts.append(paragraph)
                # print(tokenized_paragraph, len(tokenized_paragraph))
                if section in tokenized_json:
                    tokenized_json[section].append(tokenized_paragraph)
                else:
                    tokenized_json[section] = [tokenized_paragraph]
                
                paragraph_embedding = tokenizer.encode(paragraph)
                input_ids.append(paragraph_embedding)

                embed_tensor = torch.tensor(paragraph_embedding)
                with torch.no_grad():
                    logits = np.zeros((embed_tensor.shape[0], 10))
                    t_idx = 0
                    while t_idx + 512 <= embed_tensor.shape[0]:
                        part = embed_tensor[t_idx:t_idx+512]
                        part = torch.unsqueeze(part, 0)
                        temp = ner_model(part.to(self.device))[0]
                        temp = temp.squeeze(0).detach().cpu().numpy()
                        logits[t_idx:t_idx+512] = temp
                        t_idx = t_idx + 512
                    part = embed_tensor[t_idx:]
                    if len(part) != 0:
                        part = torch.unsqueeze(part, 0)
                        temp =  ner_model(part.to(self.device))[0]
                        logits[t_idx:] = temp.squeeze(0).detach().cpu().numpy()
                    predictions = np.argmax(logits, axis=-1)
                    
                    ids_set = []
                    entity_ids = [-1] * len(predictions)
                    current_entity = ''
                    item_list = []
                    for i in range(len(predictions)):
                        current_item = inverse_entity_type_dict[predictions[i]]
                        item_list.append(current_item)
                        if current_item[0] == 'B':
                            if current_entity == '':
                                current_entity = tokenized_paragraph[i]
                                entity_ids[i] = len(self.entities)
                            else:
                                
                                ids_set.append(len(self.entities))
                                self.entities.append(current_entity)
                                self.entity_types.append(current_item[2:])
                                entity_ids[i] = len(self.entities)
                                current_entity = tokenized_paragraph[i]
                        elif current_item[0] == 'O':
                            if current_entity != '':
                                ids_set.append(len(self.entities))
                                self.entities.append(current_entity)
                                self.entity_types.append(inverse_entity_type_dict[predictions[i-1]][2:])
                                current_entity = ''
                            if tokenized_paragraph[i][0] in numbers:
                                entity_ids[i] = len(self.entities)
                                ids_set.append(len(self.entities))
                                self.entities.append(tokenized_paragraph[i])
                                self.entity_types.append('score')
                                current_entity = ''
                        elif current_item[0] == 'I':
                            current_entity = current_entity + ' ' + tokenized_paragraph[i]
                            entity_ids[i] = entity_ids[i-1]
                        elif current_item[0] == 's':
                            entity_ids[i] = len(self.entities)
                            ids_set.append(len(self.entities))
                            self.entities.append(tokenized_paragraph[i])
                            self.entity_types.append('score')
                    self.entity_ids_set.append(ids_set)
                    self.entity_ids.append(entity_ids)
                    self.section_ids.append([section_id] * len(entity_ids))
                    # print(embed_tensor.shape, len(entity_ids))
                    # input()

        self.text_data_section_numbers = len(self.parts)
        table_origin = []
        column_ids = []
        row_ids = []
        entity_id = []
        table_lens = []
        caption_ids_set = []
        table_ids_set = []
        for i in range(len(table_seq)):
            try:
                caption = table_caption[i]
            except:
                print(table_caption, table_seq, i, len(table_caption), len(table_seq))
            caption_len_origin = len(caption.split(' '))
            l = 0
            entity_ids = []
            column_id = []
            row_id = []
            table_origin = []
            for j in range(len(table_seq[i])):
                for k in range(len(table_seq[i][j])):
                    table_origin = table_origin + [table_seq[i][j][k].lower()]
                    column_id = column_id + [k]
                    row_id = row_id + [j]
                    entity_ids = entity_ids + [len(self.entities)]
                    self.entities.append(table_seq[i][j][k])
                    l += 1
                    # print(table_seq[i], table_seq[i][j], table_seq[i][j][k])
                    # input()
                    if len(table_seq[i][j][k]) != 0:
                        if table_seq[i][j][k][0] in numbers:
                            self.entity_types.append('score')
                        else:
                            self.entity_types.append('O')
                    else:
                        self.entity_types.append('O')
            table_lens.append(l)
            # self.entity_ids.append([-1] + entity_ids + [-1])
            row_ids.append(row_id)
            column_ids.append(column_id)
            entity_id.append(entity_ids)
            # print(len(table_origin), len(row_id), len(column_id))
            caption_bert = ['[CLS]'] + tokenizer.tokenize(caption) + ['[SEP]']
            table_origin = caption.split(' ') + table_origin
            entity_ids_origin = [-1] * len(caption.split(' ')) + entity_ids
            # print(len(table_origin), len(entity_ids_origin))
            row_ids_origin = [-1] * len(caption.split(' ')) + row_id
            column_ids_origin = [-1] * len(caption.split(' ')) + column_id
            table = caption + ' [SEP] ' + ' '.join(table_origin)
            table_bert = ['[CLS]'] + tokenizer.tokenize(table) + ['[SEP]']
            # input()
            self.parts.append(' '.join(table_origin))

            embedding = tokenizer.encode(table)

            table_ner_tokenizer = BertTokenizer.from_pretrained('./saved_model/BERT-base-new-0-03', do_lower_case=False)
            table_ner_model = BertForTokenClassification.from_pretrained('./saved_model/BERT-base-new-0-03', num_labels=6)
            
            # table_ner_tokenizer = BertTokenizer.from_pretrained('./sa('./ved_model/BERT-base-new-{}-03-bio'.format(args.seed), do_lower_case=False)
            # table_ner_model = BertForTokenClassification.from_pretrainedsaved_model/BERT-base-new-{}-03-bio'.format(args.seed), num_labels=10)
            
            table_ner_model.to(self.device)
            table_ner_model.eval()
            entity_type_dict = {"O":0, "Material":1, "Method":2, "Metric":3, "Task":4, "score":5}
            inverse_entity_type_dict = {0:"O", 1:"Material", 2:"Method", 3:"Metric", 4:"Task", 5:"score"}
            # entity_type_dict = {"O":0, "B-Material":1, "I-Material":2, "B-Method":3, "I-Method":4, "B-Metric":5, "I-Metric":6, "B-Task":7, "I-Task":8, "score":9}
            # inverse_entity_type_dict = {0:"O", 1:"B-Material", 2:"I-Material", 3:"B-Method", 4:"I-Method", 5:"B-Metric", 6:"I-Metric", 7:"B-Task", 8:"I-Task", 9:"score"}
            with torch.no_grad():
                embed_tensor = torch.tensor(embedding)
                logits = np.zeros((embed_tensor.shape[0], 6))
                t_idx = 0
                while t_idx + 512 <= embed_tensor.shape[0]:
                    part = embed_tensor[t_idx:t_idx+512]
                    part = torch.unsqueeze(part, 0)
                    temp = table_ner_model(part.to(self.device))[0]
                    temp = temp.squeeze(0).detach().cpu().numpy()
                    logits[t_idx:t_idx+512] = temp
                    t_idx = t_idx + 512
                part = embed_tensor[t_idx:]
                if len(part) != 0:
                    part = torch.unsqueeze(part, 0)
                    temp =  table_ner_model(part.to(self.device))[0]
                    logits[t_idx:] = temp.squeeze(0).detach().cpu().numpy()
                predictions = np.argmax(logits, axis=-1)
                entity_table_norm = []
                for ii in range(len(predictions)):
                    entity_table_norm.append(inverse_entity_type_dict[predictions[ii]])
                # print(entity_norm)
            input_ids.append(embedding)
            if len(table_bert) != len(embedding):
                print(len(table_bert), len(embedding))
            
            # print(len(row_ids_origin), len(entity_ids_origin), len(column_ids_origin))

            # Normalization with the package tokenizations
            a2b, b2a = tokenizations.get_alignments(table_origin, table_bert)
            entity_origin_new = []
            for ii in range(len(a2b)):
                if a2b[ii] == []:
                    entity_origin_new.append('O')
                else:
                    entity_origin_new.append(entity_table_norm[a2b[ii][0]])
            entity_loc = []
            # for i in range(len(entity_origin)):
            # print(table_origin)
            # print(table_origin[:caption_len_origin])
            # input()
            for ii in range(caption_len_origin):
                if entity_origin_new[ii] != 'O':
                    self.entities.append(table_origin[ii])
                    self.entity_types.append(entity_origin_new[ii])
                    # self.entity_coreferences.append('')
                    entity_loc.append(ii)
                    entity_ids_origin[ii] = len(self.entities) - 1
                    # coref_origin[i] = ''
                    # num_entities += 1

            # for i in range(len(entity_origin_new) - caption_len_origin):
            #     if entity_origin[i] == 'O':
            #         if table_origin[i+caption_len_origin][0] in number_list:
            #             entity_origin_new[i+caption_len_origin] = 'score'
            #             coref_origin[i+caption_len_origin] = ''
            #         else:
            #             entity_origin_new[i+caption_len_origin] = 'O'
            #             coref_origin[i+caption_len_origin] = ''

            #         self.entities.append(table_origin[i+caption_len_origin])
            #         self.entity_types.append(entity_origin_new[i+caption_len_origin])
            #         self.entity_coreferences.append(coref_origin[i+caption_len_origin])
            #         entity_loc.append(i+caption_len_origin)
            #         entity_ids_origin[i+caption_len_origin] = num_entities
            #         num_entities += 1

            # print(table_origin)
            new_entity_ids = entity_ids_origin

            # entity_norm = []
            # coref_norm = []
            entity_ids_norm = []
            row_ids_norm = []
            column_ids_norm = []
            for ii in range(len(b2a)):
                if b2a[ii] == []:
                    # entity_norm.append('O')
                    # coref_norm.append('')
                    entity_ids_norm.append(-1)
                    # section_ids_norm.append(-1)
                    row_ids_norm.append(-1)
                    column_ids_norm.append(-1)
                else:
                    # entity_norm.append(entity_origin_new[b2a[i][0]])
                    # coref_norm.append(coref_origin[b2a[i][0]])
                    # print(len(b2a))
                    # print(b2a)
                    # print(len(entity_ids_origin))
                    # input()
                    # try:
                    entity_ids_norm.append(entity_ids_origin[b2a[ii][0]])
                    row_ids_norm.append(row_ids_origin[b2a[ii][0]])
                    column_ids_norm.append(column_ids_origin[b2a[ii][0]])
                    # except:
                    #     print(len(b2a))
                    #     print(b2a)
                    #     print(len(entity_ids_origin), len(row_ids_origin), len(column_ids_origin))
                    #     input()
                    # section_ids_norm.append(section_ids_origin[b2a[i][0]])
                    # row_ids_norm.append(row_ids_origin[b2a[i][0]])
                    # column_ids_norm.append(column_ids_origin[b2a[i][0]])
            self.entity_ids.append(entity_ids_norm)
            caption_ids_set.append(entity_ids_origin[:caption_len_origin])
            table_ids_set.append(entity_ids_origin[caption_len_origin:])
            # print(caption)
            # print(entity_ids_origin[:caption_len_origin])
            # print(len(table_origin), len(entity_ids_norm), len(embedding))
            # print(entity_ids_norm)
            # print(table_bert)
            # input()
            # new_origin_ids = section_ids_origin
            # new_words = new_words + table_origin
            table_ids_set.append(entity_ids_origin)

        self.cooccur_graph = torch.zeros((len(self.entities), len(self.entities)))
        self.align_graph = torch.zeros((len(self.entities), len(self.entities)))

        for ii in range(len(self.entity_ids_set)):
            temp_entity_ids = self.entity_ids_set[ii]
            idxs = [[ii, jj] for ii in temp_entity_ids for jj in temp_entity_ids if (ii!=-1) and (jj!=-1) and (ii!=jj) ]
            if idxs != []:
                idxs = np.array(idxs)
                self.cooccur_graph[idxs[:,0], idxs[:,1]] = 1
        
        for ii in range(len(table_seq)):
            caption_ids = list(set(caption_ids_set[ii]))[1:]
            # print(column_ids)
            for iii in range(len(column_ids[ii])):
                for jjj in range(len(column_ids[ii])):
                    # try:
                    if (((row_ids[ii][iii] == row_ids[ii][jjj]) or (column_ids[ii][iii] == column_ids[ii][jjj]))) and ((self.entity_types[entity_id[ii][iii]] != 'score' and self.entity_types[entity_id[ii][jjj]] == 'score') or (self.entity_types[entity_id[ii][iii]] == 'score' and self.entity_types[entity_id[ii][jjj]] != 'score')):
                # if (((row_ids_origin[ii] == row_ids_origin[jj]) or (column_ids_origin[ii] == column_ids_origin[jj])) and ((entity_ids_origin[ii] != -1) and (entity_ids_origin[jj] != -1))) and ((self.entity_types[entity_ids_origin[ii]] != 'score' and self.entity_types[entity_ids_origin[jj]] == 'score') or (self.entity_types[entity_ids_origin[ii]] == 'score' and self.entity_types[entity_ids_origin[jj]] != 'score')):
                        self.align_graph[entity_id[ii][iii], entity_id[ii][jjj]] = 1
                    # except:
                    #     print(self.entity_types)
                    #     print(len(self.entity_types), len(self.entities))

    
        return input_ids, tokenized_json

    def find_paragraph_ids(self, tokenized_text_json, words):
        paragraph_ids = [0] * len(words)
        start = 0
        idx = 0
        for section in tokenized_text_json:
            for paragraph in tokenized_text_json[section]:
                paragraph_ids[start:start+len(paragraph)] = [idx] * len(paragraph)
                start += len(paragraph)
                idx += 1
        # Annotating tables
        for l in table_lens:
            paragraph_ids[start:start+l+1] = [idx] * (l)
            start = start + l + 1
        return words, paragraph_ids

    def read_tuple_labels(self, args, tuples, words, tokenizer, BERTmodel):
        self.gt_tuples = []
        self.questions = []
        self.para_labels = []
        self.question_ids = []
        self.tuple_embedding = []
        self.elements_embedding = []
        with torch.no_grad():
            for relation in tuples:
                if relation[3] != '-':
                    question = "What is the score for " + relation[2] + " metric on " + relation[1] + " dataset for " + relation[0] + " task?" 
                    question_idx = np.full(len(question.split()), -1)
                    metric_len = len(relation[2].split())
                    dataset_len = len(relation[1].split())
                    task_len = len(relation[0].split())
                    question_idx[5:5+metric_len] = 2
                    question_idx[8+metric_len-1:8+metric_len+dataset_len-1] = 1
                    question_idx[11+metric_len+dataset_len-2:11+metric_len+dataset_len+task_len-2] = 0
                    # Might need to change to include tables
                    # flag = False
                    # for i in range(len(self.parts)):
                    #     print(self.parts[i])
                    #     print(relation[3])
                    #     input()
                    #     if relation[3] in self.parts[i]:
                    #         flag = True
                    if relation[3] in words:
                        para_labels = []
                        self.gt_tuples.append(tuple(relation))
                        self.questions.append(question)
                        for i in range(len(self.parts)):
                            if relation[3] in self.parts[i]:
                                para_labels.append(i)
                        if para_labels != []:
                            self.para_labels.append(para_labels)
                            if args.query_style == 'question':
                                question_origin = question.split()
                                question_bert = ['[CLS]'] + tokenizer.tokenize(question) + ['[SEP]']
                                _, b2a = tokenizations.get_alignments(question_origin, question_bert)
                                question_idx_norm = []
                                for i in range(len(b2a)):
                                    if b2a[i] == []:
                                        question_idx_norm.append(-1)
                                    else:
                                        question_idx_norm.append(question_idx[b2a[i][0]])
                                question_id = torch.tensor(tokenizer.encode(question))
                                self.question_ids.append(question_id)
                                question_id = torch.unsqueeze(question_id, 0)

                                question_embed = BERTmodel(question_id.to(args.device)).last_hidden_state.cpu()
                                self.tuple_embedding.append(question_embed.squeeze(0).cpu()[0])
                                
                                elements_embedding = torch.zeros((len(relation)-1, question_embed.shape[-1]))
                                
                                num = torch.zeros(len(relation) - 1)
                                question_embed = question_embed.squeeze(0)
                                for i in range(len(question_embed)):
                                    if question_idx_norm[i] != -1:
                                        elements_embedding[int(question_idx_norm[i])] += question_embed[i].cpu()
                                        num[int(question_idx_norm[i])] += 1
                                for i in range(len(elements_embedding)):
                                    elements_embedding[i] = elements_embedding[i] / (
                                                num[i] * torch.ones_like(elements_embedding[i]))
                                self.elements_embedding.append(elements_embedding)
                            torch.cuda.empty_cache()
                            if args.duplicate == 'n':
                                break
                # self.para_labels.append(para_labels)
    
    def BERT_encoder(self, args, input_ids, BERTmodel):
        if args.saved_embed == 'n':
            # BERT encode the sentences
            embeddings = []
            self.para_embeddings = []
            self.para_entity_embeddings = []
            with torch.no_grad():
                for input_id in input_ids:
                    if len(input_id) > 512:
                        idx = 0
                        input_id = torch.tensor(input_id)
                        embedding = torch.zeros((len(input_id), 768))
                        while idx + 512 <= len(input_id):
                            part = input_id[idx:idx+512]
                            part = torch.unsqueeze(part, 0)
                            embedding[idx:idx+512] = BERTmodel(part.to(args.device)).last_hidden_state.cpu()
                            idx = idx + 512
                        part = input_id[idx:]
                        if len(part) != 0:
                            part = torch.unsqueeze(part, 0)
                            embedding[idx:] = BERTmodel(part.to(args.device)).last_hidden_state.cpu()
                        embeddings.append(embedding)
                    else:
                        input_id = torch.tensor(input_id)
                        input_id = torch.unsqueeze(input_id, 0)
                        embedding = BERTmodel(input_id.to(args.device)).last_hidden_state.cpu()
                        embeddings.append(embedding.squeeze(0))
                    self.para_embeddings.append(embedding.squeeze(0)[0])
                    self.para_entity_embeddings.append(embedding.squeeze(0))
                    torch.cuda.empty_cache()
            # Compute the mention embeddings
            num = torch.zeros(len(self.entities))
            self.initial_embed = torch.zeros((len(self.entities), 768))
            
            for i in range(len(self.entity_ids)):
                for j in range(len(self.entity_ids[i])):
                    if (self.entity_ids[i][j] != -1):
                        self.initial_embed[self.entity_ids[i][j]] += embeddings[i][j]
                        num[self.entity_ids[i][j]] += 1

            for i in range(len(self.entities)):
                if num[i] == 0:
                    num[i] = 1
                    # self.initial_embed[i] = torch.ones(768)
            
            for i in range(self.initial_embed.shape[0]):
                self.initial_embed[i] = self.initial_embed[i] / (num[i] * torch.ones_like(self.initial_embed[i]))



def read_files(args, text_path, table_path, tuple_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    BERTmodel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BERTmodel.to(device)
    BERTmodel.eval()
    print("Reading data...")

     # For displaying info
    valid_tuples = 0
    num_tuples = 0
    invalid_doc = 0
    text_ans_number = 0
    table_ans_number = 0
    count = 0

    filename_list = []
    text_list = []
    document_list = []
    tuple_dict = {}

    # Parse document text files
    for filename in os.listdir(text_path):
        with open(os.path.join(text_path, filename), 'r') as text_file:
            filename = os.path.splitext(filename)[0]
            filename_list.append(filename)
            text_list.append(parse_text_to_json(text_file))

    # Parse annotation result files
    with open(tuple_path, 'r') as tuple_file:
        tuple_dict = parse_tuple_to_dict(tuple_file)

    for i in range(len(filename_list)):
        filename = filename_list[i]
        text_json = text_list[i]
        table_json = {}
        table_caption = []

        table_json_path = os.path.join(table_path, filename + '.json')
        if os.path.isfile(table_json_path):
            with open(table_json_path, 'r') as table_file:
                table_json, table_caption = reconstruct_table(json.load(table_file))
                table_seq = flatten_ary(flatten_ary(table_json))
        
        # print(table_caption)
        # print(table_seq)
        # input()
        # if len(table_caption) != len(table_seq):
        #     print('ERROR')
        #     print(table_caption, table_seq)
        #     input()

        if filename in tuple_dict:
            tuples = tuple_dict[filename]
            
            single_document = TDMSdataloader(filename, text_json, table_seq, table_caption, tuples, tokenizer, BERTmodel, args)
            
            try:
                if len(single_document.tuple_embedding) != len(single_document.para_labels):
                    print('error')
                if single_document.gt_tuples != [] or single_document.tuple_embedding != []:
                    document_list.append(single_document)
                else:
                    invalid_doc += 1
            except:
                invalid_doc += 1
        else:
            invalid_doc += 1

        # Display document info
        print('Reading document #' + str(count))
        count += 1
        # if count == 5:
        #     return document_list
    
    print("There are {} documents are invalid.".format(invalid_doc))
    print('Reading data Done!!!')
    print('===============================================================================')
    
    return document_list


def link_start(args):
    if args.saved_embed == 'n':
        time_start = time.time()
        pdf_txt_path = './datasets/TDMS/dataset/pdfFile_txt'
        pdf_table_path = './datasets/TDMS/dataset/pdfFile_table'
        tdms_tuple_path = './datasets/TDMS/dataset/resultsAnnotation.tsv'
        train_dev_list = read_files(args, pdf_txt_path, pdf_table_path, tdms_tuple_path)
        time_end = time.time()
        print('Time cost: ',time_end - time_start,'s')

        torch.save(train_dev_list, './saved_model/train_dev_test_tdms.pt')
    else:
        train_dev_list = torch.load('./saved_model/train_dev_test_tdms.pt')
    return train_dev_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.002, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-9, type=float, help="Parameter Epsilon for AdamW optimizer.")
    parser.add_argument("--epochs", default=10, type=int, help="The number of the epochs.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="The max value of the gradient normalization.")
    # parser.add_argument("--bert_model", default='base', type=str, help="The pretrained model.")
    parser.add_argument("--use_entity_type", action="store_true", help="Whether use entity_type or not.")
    parser.add_argument("--use_entity_ids", action="store_true", help="Whether use entity_ids or not.")
    parser.add_argument("--entity_type_embed_size", default=768, type=int, help="The size of the entity type embedding.")
    parser.add_argument("--saved_embed", default='n', type=str, help="Use saved embedding or not.")
    parser.add_argument("--gcn_layers", default=1, type=int, help="The number of the GCN layers.")
    parser.add_argument("--focus", default='score', type=str, help="The query element in the tuple.")
    parser.add_argument("--embed_style", default='paragraphs', type=str, help="The pretrained model.")
    parser.add_argument("--table_style", default='table', type=str, help="The table embedding style [table/caption/caption+table].")
    parser.add_argument("--partial", action="store_true", help="Use parts of the documents as candidate pools.")
    parser.add_argument("--query_style", default='question', type=str, help="The query embedding style [tuple/question].")
    parser.add_argument("--duplicate", default='y', type=str, help="Allow duplication of tuples in a single document or not.")

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    set_seed(args)

    if args.saved_embed == 'n':
        time_start = time.time()
        pdf_txt_path = './datasets/TDMS/dataset/pdfFile_txt'
        pdf_table_path = './datasets/TDMS/dataset/pdfFile_table'
        tdms_tuple_path = './datasets/TDMS/dataset/resultsAnnotation.tsv'
        train_dev_list = read_files(args, pdf_txt_path, pdf_table_path, tdms_tuple_path)
        time_end = time.time()
        print('Time cost: ',time_end - time_start,'s')

        torch.save(train_dev_list, './datasets/saved_model/train_dev_test_tdms.pt')
    else:
        train_dev_list = torch.load('./datasets/saved_model/train_dev_test_tdms.pt')

if __name__ == '__main__':
    main()