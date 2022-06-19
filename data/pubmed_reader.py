import json
from turtle import pu
import jsonlines
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, dataloader, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer
import tokenizations
from keras.preprocessing.sequence import pad_sequences
from itertools import product
import random
import time
import csv
import json
import argparse
from high_level_utils import *
# from retriever import *
# from rl_bfs import BFS


number_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

class BERTdataloader(object):
    def __init__(self, tokenizer, BERTmodel, item, idx, args, reference_dict, ll):
        super(BERTdataloader, self).__init__()

        self.id = item['pmid']
        relations = item['triple_candidates']
        flag = False
        self.tup_num = 0
        # for rel in relations:
        #     label = rel["label"]
        #     if label == 1:
        flag = True


        # read tables
        
        # table_caption = self.caption_reader(self.id)
        # tables, table_lens, table_captions, table_caption_indexes = self.table_caption_reader(self.id)
        # self.num_tables = len(tables)
        # origin_tables, origin_table_lens, origin_table_indexes = self.table_reader(self.id)
        # tables, table_lens, table_captions, table_caption_indexes = self.table_caption_alignment(tables, table_lens, table_captions, table_caption_indexes, origin_tables, origin_table_lens, origin_table_indexes)

        # read reference list
        # reference_list = reference_dict[self.id]
        if flag:
            # Yingjun 10/30: words changed to paragraphs, and flatten the list of paragraphs
            words = [word.lower() for paragraph in item['paragraphs'] for word in paragraph]
            words_idx = len(words)
            # if tables != []:
            #     words = words + tables
            # Yingjun 10/30: changed to drug, gene, variant triple
            self.entity_type_dict = {"drug":0, "gene":1, "variant":2}

            # # Record the paragraph labels
            table_lens = []
            paragraph_ids = self.read_paragraph_labels(item, words, words_idx, table_lens)

            # Read the NER labels from the dataset
            paras = item['paragraphs']
            parts = []
            startp = 0
            endp = 0
            for i in range(len(paras)):
                endp = startp + len(paras[i]) - 1
                parts.append([startp, endp])
                startp = startp + len(paras[i])

            origin_ids = paragraph_ids
            self.text_data_section_numbers = len(parts)

            ner_results, entity_ids, entity_types, num_entities, coreference = self.read_ner_labels(item, words, parts, origin_ids)
            # print(words[:50])
            # print(self.entities[:50])
            # print(self.entity_types[:50])
            # print(num_entities)
            # input()

            

            # add scores to the dataset
            # num_entities = len(self.ner)
            # print(self.ner)
            # print(num_entities)
            # ner_results, entity_ids, entity_types, num_entities = self.annotate_scores(words, ner_results, entity_ids, entity_types)

            
            # coreference = self.read_coreference_labels(words, item, parts, table_lens, num_entities, entity_ids, origin_ids)

            # # create_reference_tables(words)
            # self.ref_graph_para = torch.zeros_like(self.coref_graph_para)

            # Normalize the tokenization from both dataset and BERT
            # input_ids = self.token_normalization(args, tokenizer, num_entities, parts, words, ner_results, coreference, entity_ids, origin_ids, words_idx, table_lens, table_captions, table_caption_indexes, reference_list)
            input_ids = self.token_normalization(args, tokenizer, num_entities, parts, words, ner_results, coreference, entity_ids, origin_ids, words_idx)
            self.input_ids = input_ids

            # record the positive n-Ary relation tuples
            flag = self.read_tuple_labels(item, words, coreference, entity_ids, origin_ids, args, tokenizer, BERTmodel, ll)
            
            # Padding the sentence and set up the sentence mask
            self.BERT_encoder(args, input_ids, BERTmodel)

           

            if len(self.entity_ids_label) != len(self.para_embeddings):
                print("Reading...", len(self.entity_ids_label), len(self.para_embeddings))

    
    # Yingjun 10/30: tailored read_ner_labels to the data structure of mentions in pubmed
    # Note that unlike ner, mentions have a list of paragraph. The idx is the local idx within the paragraph
    def read_ner_labels(self, item, words, parts, origin_ids):
        # Record the ner labels
        self.entities = []
        self.ner = item['mentions']
        ner = item['mentions']
        ner_results = ['O'] * len(words)
        self.entity_types = []
        entity_ids = [-1] * len(words)
        coreference = [''] * len(words)
        self.entity_coreferences = []
        entity_types = [[], [], []]

        num_ner = 0
        # be careful the list of paragraphs have been flattened under words
        example_all_mentions = dict()
        example_para_all_mentions = dict()
        paragraph_first_idx = 0
        for paragraph_idx in range(len(ner)):
            paragraph_mentions = ner[paragraph_idx]
            for local_idx in range(len(paragraph_mentions)):
                # locate the span of entity
                curr_mention = paragraph_mentions[local_idx]
                ment_start, ment_end, ment_type, ment_word = \
                    curr_mention['start'],curr_mention['end'],curr_mention['type'],curr_mention['name']

                global_start_idx =  paragraph_first_idx + ment_start
                global_end_idx = global_start_idx + (ment_end - ment_start)
                # 1. the actual word of entity
                entity = words[global_start_idx:global_end_idx]
                # if self.id == '17184525':
                #     print(words[global_start_idx:global_end_idx])
                #     print(ner_results[global_start_idx:global_end_idx])
                #     print(coreference[global_start_idx:global_end_idx])
                #     print(ment_type, ment_word)
                #     input()
                if coreference[global_start_idx] == '':
                    self.entities.append(entity)
                    # 2. type of ner
                    ner_results[global_start_idx:global_end_idx] = [ment_type] * (global_end_idx - global_start_idx)
                    self.entity_types.append(ment_type)
                    coreference[global_start_idx:global_end_idx] = [ment_word] * (global_end_idx - global_start_idx)
                    self.entity_coreferences.append(ment_word)
                    # 3. how many ners have we seen and collected so far?
                    entity_ids[global_start_idx:global_end_idx] = [num_ner] * (global_end_idx - global_start_idx)
                    num_ner += 1
                    # 4. append curr word idx to the list associated with ment_type
                    entity_types[int(self.entity_type_dict[ment_type])].append(global_start_idx)

                if ment_word not in example_all_mentions:
                    example_all_mentions[ment_word] = []
                if ment_word not in example_para_all_mentions:
                    example_para_all_mentions[ment_word] = []
                example_all_mentions[ment_word].append(entity_ids[global_start_idx])
                example_para_all_mentions[ment_word].append(origin_ids[global_start_idx])
            paragraph_first_idx += len(item['paragraphs'][paragraph_idx])

        num_entities = num_ner
        self.coref_graph = torch.zeros((num_entities, num_entities))
        self.coref_graph_para = torch.zeros((len(parts), len(parts)))
        for ment_key in example_all_mentions:
            idx = [[i, j] for i in example_all_mentions[ment_key] for j in example_all_mentions[ment_key]]
            idx_para = [[i, j] for i in example_para_all_mentions[ment_key] for j in example_para_all_mentions[ment_key]]
            if idx != []:
                idx = np.array(idx)
                self.coref_graph[idx[:, 0], idx[:, 1]] = 1
                idx_para = np.array(idx_para)
                self.coref_graph_para[idx_para[:, 0], idx_para[:, 1]] = 1

            
        return ner_results, entity_ids, entity_types, num_ner, coreference


    # Yingjun 10/30: add read_paragraph_labels function following Yuchen's previous codes
    '''
    input: idx of word
    return: idx of paragraph which this word belongs to
    '''
    def read_paragraph_labels(self, item, words, words_idx, table_lens):
        paragraph = item['paragraphs']
        paragraph_ids = [0] * len(words)
        start = 0
        for idx in range(len(paragraph)):
            paragraph_ids[start:start + len(paragraph[idx])] = [idx] * len(paragraph[idx])
            start += len(paragraph[idx])

        idx_pointer = words_idx
        for idx in range(len(table_lens)):
            paragraph_ids[idx_pointer:idx_pointer + table_lens[idx]] = [len(paragraph) + idx] * (table_lens[idx])
            idx_pointer += table_lens[idx]
        return paragraph_ids


    # Yingjun 10/30: re-write read_corefrence_labels to according to item['mentions']
    # def read_coreference_labels(self, words, item, parts, table_lens, num_entities, entity_ids, origin_ids):
    #     coreference = [''] * len(words)
    #     # self.entity_coreferences = [''] * len(self.entities)
    #     coref = item['mentions']
    #     self.coref = coref
    #     self.coref_graph = torch.zeros((num_entities, num_entities))
    #     self.coref_graph_para = torch.zeros((len(parts)+len(table_lens), len(parts)+len(table_lens)))
    #     for i in range(num_entities):
    #         self.coref_graph[i, i] = 1

    #     example_all_mentions = dict()
    #     example_para_all_mentions = dict()
    #     paragraph_first_idx = 0
    #     for paragraph_idx in range(len(coref)):
    #         paragraph_mentions = coref[paragraph_idx]
    #         for ment in paragraph_mentions:
    #             ment_start, ment_end, ment_type, ment_word = \
    #                 ment['start'], ment['end'], ment['type'], ment['name']

    #             global_start_idx = paragraph_first_idx + ment_start
    #             global_end_idx = global_start_idx + (ment_end - ment_start)
    #             coreference[global_start_idx:global_end_idx] = [ment_word] * (global_end_idx - global_start_idx)
    #             # each mention has one example list and an example_para list
    #             if ment_word not in example_all_mentions:
    #                 example_all_mentions[ment_word] = []
    #             if ment_word not in example_para_all_mentions:
    #                 example_para_all_mentions[ment_word] = []
    #             example_all_mentions[ment_word].append(entity_ids[global_start_idx])
    #             example_para_all_mentions[ment_word].append(origin_ids[global_start_idx])

    #             for ment_key in example_all_mentions:
    #                 idx = [[i, j] for i in example_all_mentions[ment_key] for j in example_all_mentions[ment_key]]
    #                 idx_para = [[i, j] for i in example_para_all_mentions[ment_key] for j in example_para_all_mentions[ment_key]]
    #                 if idx != []:
    #                     idx = np.array(idx)
    #                     self.coref_graph[idx[:, 0], idx[:, 1]] = 1
    #                     idx_para = np.array(idx_para)
    #                     self.coref_graph_para[idx_para[:, 0], idx_para[:, 1]] = 1
    #                 # for i in range(len(example_all_mentions[ment_key])):
    #                 #     self.entity_coreferences[example_all_mentions[ment_key][i]] = ment_key
    #     return coreference

    # def token_normalization(self, args, tokenizer, num_entities, parts, words, ner_results, coreference, entity_ids, origin_ids, words_idx, table_lens, table_captions, caption_ids, reference_list):
    def token_normalization(self, args, tokenizer, num_entities, parts, words, ner_results, coreference, entity_ids, origin_ids, words_idx):
        self.entity_label = []
        self.coref_label = []
        self.entity_ids_label = []
        self.section_ids_label = []
        input_ids = []
        total_num = len(parts) #+ len(table_captions)
        self.reference_table = torch.zeros((total_num, total_num))
        self.cooccur_graph = torch.zeros((num_entities, num_entities))
        for i in range(num_entities):
            self.cooccur_graph[i, i] = 1
        self.parts = []
        self.parts_bert = []
        self.text_caption = []
        idx_num = 0
        idx_ref = 0
        # print(parts)
        for idx in parts:
            sentence_origin = words[idx[0]:idx[1]]
            entity_origin = ner_results[idx[0]:idx[1]]
            coref_origin = coreference[idx[0]:idx[1]]
            entity_ids_origin = entity_ids[idx[0]:idx[1]]

            section_ids_origin = origin_ids[idx[0]:idx[1]]
            temp_sentence = sentence_origin

            sentence = ' '.join(sentence_origin)
            
            idx_num += 1

            sentence_bert = ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]']
            self.parts.append(sentence)
            self.parts_bert.append(sentence_bert)
            self.text_caption.append(sentence)

            # Use the bert tokenizer to tokenize the sentences
            # self.sentences.append(sentence)
            embedding = tokenizer.encode(sentence)
            input_ids.append(embedding)
            if len(sentence_bert) != len(embedding):
                print(len(sentence_bert), len(embedding))

            # Normalization with the package tokenizations
            a2b, b2a = tokenizations.get_alignments(sentence_origin, sentence_bert)
            entity_norm = []
            coref_norm = []
            entity_ids_norm = []
            section_ids_norm = []
            for i in range(len(b2a)):
                if b2a[i] == []:
                    entity_norm.append('O')
                    coref_norm.append('')
                    entity_ids_norm.append(-1)
                    section_ids_norm.append(-1)
                else:
                    entity_norm.append(entity_origin[b2a[i][0]])
                    coref_norm.append(coref_origin[b2a[i][0]])
                    entity_ids_norm.append(entity_ids_origin[b2a[i][0]])
                    section_ids_norm.append(section_ids_origin[b2a[i][0]])
            self.entity_label.append(entity_norm)
            self.coref_label.append(coref_norm)
            self.entity_ids_label.append(entity_ids_norm)
            self.section_ids_label.append(section_ids_norm)
            entity_ids_norm = list(set(entity_ids_norm))
        
        for idx in parts:
            sentence_origin = words[idx[0]:idx[1]]
            entity_ids_origin = entity_ids[idx[0]:idx[1]]
            while '.' in sentence_origin:
                temp_idx = sentence_origin.index('.') + 1
                temp_entity_ids = entity_ids_origin[:temp_idx]
                idxs = [[i, j] for i in temp_entity_ids for j in temp_entity_ids if (i!=-1) and (j!=-1) and (i!=j) ]
                if idxs != []:
                    idxs = np.array(idxs)
                    self.cooccur_graph[idxs[:,0], idxs[:,1]] = 1
                sentence_origin = sentence_origin[temp_idx:]
                entity_ids_origin = entity_ids_origin[temp_idx:]
        
        return input_ids

    # Yingjun 10/30: re-write read_tuple_labels according to item['triple_candidates']
    def read_tuple_labels(self, item, words, coreference, entity_ids, origin_ids, args, tokenizer, BERTmodel, ll):
        relations = item['triple_candidates']
        # No more method_subrelations
        # methods = item['method_subrelations']
        self.gt_tuples = []
        self.questions = []
        self.tuple_embedding = []
        # self.question_embedding = []
        self.elements_embedding = []
        self.tup_num = 0
        self.tup_label = []
        self.tup_candidate = []
        self.para_labels = []
        tup_len = []
        self.question_ids = []
        content = words
        # print(len(entity_ids), len(origin_ids))
        # input()
        flag = False
        with torch.no_grad():
            for rel in relations:
                if args.focus == 'drug':
                    tup = [rel["variant"], rel["gene"], rel["drug"]]
                    question = "What is the drug related to " + rel["gene"] + " gene and " + rel[
                        "variant"] + " variant?"
                    tup_idx = [0, 1, 2]
                    question_idx = np.ones(len(question.split(" "))) * (-1)
                    question_idx[6] = 1
                    question_idx[9] = 2
                elif args.focus == 'gene':
                    tup = [rel["drug"], rel["variant"], rel["gene"]]
                    question = "What is the gene related to " + rel["drug"] + " drug and " + rel[
                        "variant"] + " variant?"
                    tup_idx = [0, 1, 2]
                    question_idx = np.ones(len(question.split(" "))) * (-1)
                    question_idx[6] = 0
                    question_idx[9] = 2
                elif args.focus == 'variant':
                    tup = [rel["drug"], rel["gene"], rel["variant"]]
                    question = "What is the variant related to " + rel["drug"] + " drug and " + rel[
                        "gene"] + " gene?"
                    tup_idx = [0, 1, 2]
                    question_idx = np.ones(len(question.split(" "))) * (-1)
                    question_idx[6] = 0
                    question_idx[9] = 1
                label = rel["label"]
                if (label == ll) and tup[-1] in coreference:
                    flag = True
                    self.gt_tuples.append(tup)
                    self.questions.append(question)
                    self.tup_num += 1
                    if not tup[-1] in coreference:
                        print(self.id)
                        print(coreference)
                        print(list(set(self.entity_coreferences)))
                        print(tup)
                        input()
                    self.tup_label.append(entity_ids[coreference.index(tup[-1])])
                    candidate_section = origin_ids[coreference.index(tup[-1])]
                    self.para_labels.append(candidate_section)
                    section_range = [entity_ids[i] for i in range(len(origin_ids)) if
                                     (origin_ids[i] == candidate_section) and (entity_ids[i] >= len(self.ner))]
                    if args.partial == True and section_range != []:
                        self.tup_candidate.append([min(section_range), max(section_range)])
                    else:
                        self.tup_candidate.append([0, len(self.entities) - 1])
                    # tup_len.append(max(section_range) - min(section_range))
                    if args.query_style == 'tuple':
                        # tup = ['[CLS]'] + tup + ['[SEP]']
                        tup_query = ' '.join(tup[:-1])
                        tup_bert = ['[CLS]'] + tokenizer.tokenize(tup_query) + ['[SEP]']
                        a2b, b2a = tokenizations.get_alignments(tup_query, tup_bert)
                        tup_idx_norm = []
                        for i in range(len(b2a)):
                            if b2a[i] == []:
                                tup_idx_norm.append(-1)
                            else:
                                tup_idx_norm.append(tup_idx[b2a[i][0]])

                        tup_query_id = tokenizer.encode(tup_query)
                        tup_query_id = torch.tensor(tup_query_id)
                        tup_query_id = torch.unsqueeze(tup_query_id, 0)
                        torch.cuda.empty_cache()
                        # print(tup_query, tup_query_id)
                        tup_embed = BERTmodel(tup_query_id.to(args.device)).last_hidden_state.cpu()
                        self.tuple_embedding.append(tup_embed.squeeze(0).cpu()[0])
                        elements_embedding = torch.zeros((len(tup) - 1, tup_embed.shape[-1]))
                        num = torch.zeros(len(tup) - 1)
                        tup_embed = tup_embed.squeeze(0)
                        for i in range(len(tup_embed)):
                            if tup_idx_norm[i] != -1:
                                # Is this an error should we use tup_idx_norm instead??
                                elements_embedding[int(question_idx_norm[i])] += tup_embed[i].cpu()
                                num[int(question_idx_norm[i])] += 1
                        for i in range(len(elements_embedding)):
                            elements_embedding[i] = elements_embedding[i] / (
                                        num[i] * torch.ones_like(elements_embedding[i]))
                        self.elements_embedding.append(elements_embedding)
                    else:
                        # question = ['[CLS]'] + question + ['[SEP]']
                        question_origin = question.split(' ')
                        question_bert = ['[CLS]'] + tokenizer.tokenize(question) + ['[SEP]']
                        a2b, b2a = tokenizations.get_alignments(question_origin, question_bert)
                        question_idx_norm = []
                        for i in range(len(b2a)):
                            if b2a[i] == []:
                                question_idx_norm.append(-1)
                            else:
                                question_idx_norm.append(question_idx[b2a[i][0]])
                        question_id = tokenizer.encode(question)
                        question_id = torch.tensor(question_id)
                        self.question_ids.append(question_id)
                        question_id = torch.unsqueeze(question_id, 0)

                        question_embed = BERTmodel(question_id.to(args.device)).last_hidden_state.cpu()
                        self.tuple_embedding.append(question_embed.squeeze(0).cpu()[0])
                        elements_embedding = torch.zeros((len(tup) - 1, question_embed.shape[-1]))
                        num = torch.zeros(len(tup) - 1)
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
        return flag
    
    
    def BERT_encoder(self, args, input_ids, BERTmodel):
        MAX_LEN = 512
        # BERTmodel = BERTmodel.to("cuda:0")

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
            
            for i in range(len(self.entity_ids_label)):
                for j in range(len(self.entity_ids_label[i])):
                    if (self.entity_label[i][j] != 'O'):
                        # print(len(self.entities), self.entity_ids_label[i][j],i,j, len(embeddings))
                        self.initial_embed[self.entity_ids_label[i][j]] += embeddings[i][j]
                        num[self.entity_ids_label[i][j]] += 1

            for i in range(len(self.entities)):
                if num[i] == 0:
                    num[i] = 1
                    self.initial_embed[i] = torch.ones(768)
            # # self.initial_embed[len(self.entities)] = torch.ones(768)
            
            for i in range(self.initial_embed.shape[0]):
                self.initial_embed[i] = self.initial_embed[i] / (num[i] * torch.ones_like(self.initial_embed[i]))
        #     torch.save(self.initial_embed, "./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "pubmed_nt_embed.pt")
        #     torch.save(self.para_embeddings, "./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "pubmed_" + args.embed_style + "_" + args.table_style + "_nt_embed.pt")
        #     torch.save(self.para_entity_embeddings, "./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "pubmed_" + args.embed_style + "_" + args.table_style + "_entity_nt_embed.pt")
        # else:
        #     self.initial_embed = torch.load("./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "pubmed_nt_embed.pt")
        #     self.para_embeddings = torch.load("./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "pubmed_" + args.embed_style + "_" + args.table_style + "_nt_embed.pt")
        #     self.para_entity_embeddings = torch.load("./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "pubmed_" + args.embed_style + "_" + args.table_style + "_entity_nt_embed.pt")

    # def compute_retrieval_dist(self):
    #     items = self.text_caption
    #     retriever = TfIdf()
    #     rdist = torch.zeros((len(items), len(items)))
    #     for idx in range(len(items)):
    #         rdist[idx] = torch.tensor(retriever.dists(items[idx], items))
    #     return rdist

    def compute_gt_paths(self, gt_paths, gt_actions):
        self.gt_paths = gt_paths
        self.gt_actions = gt_actions
        
    def compute_sim_scores(self, args):
        self.input_size = 8
        self.scores = []
        for tup_idx in range(len(self.gt_tuples)):
            adjacent_graph = self.coref_graph + self.cooccur_graph
            
            entity_ids_set = []
            for i in range(len(self.entity_ids_label[self.para_labels[tup_idx]])):
                if self.entity_ids_label[self.para_labels[tup_idx]][i] != -1 and self.entity_ids_label[self.para_labels[tup_idx]][i] not in entity_ids_set and self.entity_types[self.entity_ids_label[self.para_labels[tup_idx]][i]] == 'score':
                    entity_ids_set.append(self.entity_ids_label[self.para_labels[tup_idx]][i])
            entity_ids_set.sort()

            elements_strings = self.gt_tuples[tup_idx][:-1]
            
            current_id = 0
            
            iterations = range(len(self.entities))
            scores = torch.zeros((len(self.entities), self.input_size))
            for i in iterations:
                # best_neighbour = [[],[],[],[]]
                # all_neighbours = []
                for j in range(adjacent_graph.shape[1]):
                    if adjacent_graph[i,j] != 0:
                        # all_neighbours.append(self.entities[j])
                        idx_k = 0
                        for k in range(len(elements_strings)):
                            current_sim = 1 - LevenDist(''.join(self.entities[j]), elements_strings[k].lower())
                            if current_sim > scores[current_id, idx_k]:
                                scores[current_id,idx_k] = current_sim
                                # best_neighbour[k] = self.entities[j]
                            idx_k += 1
                            current_sim = LCSSim(''.join(self.entities[j]), elements_strings[k].lower())
                            if current_sim > scores[current_id, idx_k]:
                                scores[current_id,idx_k] = current_sim
                                # best_neighbour[k] = self.entities[j]
                            idx_k += 1
                            current_sim = LCSSim2(''.join(self.entities[j]), elements_strings[k].lower())
                            if current_sim > scores[current_id, idx_k]:
                                scores[current_id,idx_k] = current_sim
                            idx_k += 1
                            icurrent_sim = F.cosine_similarity(self.initial_embed[j].unsqueeze(0), self.elements_embedding[tup_idx][k].unsqueeze(0), dim=-1)
                            if current_sim > scores[current_id, idx_k]:
                                scores[current_id,idx_k] = current_sim
                            idx_k += 1
                current_id += 1
            self.scores.append(scores)

def read_references():
    reference_dict = None
    flag = True
    file_path = './datasets/scirex/ref_dict_0921_no_unk_tab.json'
    print("Reading references...")
    with open(file_path, 'r') as infile:
        for item in jsonlines.Reader(infile):
            if flag:
                reference_dict = item
                flag = False
            else:
                reference_dict.update(item)
    return reference_dict

def read_files(args, file_path):
    document_list = []
    idx = 0
    if args.bert_model == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        BERTmodel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    elif args.bert_model == 'sci':
        tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        BERTmodel = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BERTmodel.to(device)
    BERTmodel.eval()
    print("Reading data...")
    valid_tuples = 0
    num_tuples = 0
    invalid_doc = 0
    reference_dict = read_references()
    text_ans_number = 0
    table_ans_number = 0
    # For displaying info (displaying current document number)
    count = 0
    if 'test' in file_path:
        ll = 0
        print(file_path, ll)
    else:
        ll = 1
        print(file_path, ll)
    with jsonlines.open(file_path, 'r') as reader:
        # jsonlines can only read line by line
        for item in reader:
            document_single = BERTdataloader(tokenizer, BERTmodel, item, idx, args, reference_dict, ll)
            # print("reading document#"+str(count))
            count += 1
            if document_single.tup_num !=0:
                document_list.append(document_single)
                valid_tuples += document_single.tup_num
                num_tuples += len(document_single.gt_tuples)
                for label in document_single.para_labels:
                    if label < document_single.text_data_section_numbers:
                        text_ans_number += 1
                    else:
                        table_ans_number += 1
            else:
                invalid_doc += 1
            idx += 1
            
            document_single.compute_sim_scores(args)
            # if args.bfs:
            #     path_list = []
            #     action_list = []
            #     for query_idx in range(len(document_single.para_labels)):
            #         path, action = BFS(document_single, document_single.para_labels[query_idx], document_single.gt_tuples[query_idx])
            #         path_list.append(path)
            #         action_list.append(action)
            #     document_single.compute_gt_paths(path_list, action_list)

    print("The total number of documents in the dataset: {}".format(len(document_list)))
    print("There are {}/{} tuples are valid.".format(valid_tuples, num_tuples))
    print("There are {} documents are invalid.".format(invalid_doc))
    print("Text answers:{}, Table answers: {}".format(text_ans_number, table_ans_number))
    print("Reading data Done!!!")
    print('===============================================================================')
    
    return document_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.002, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    # parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-9, type=float, help="Parameter Epsilon for AdamW optimizer.")
    parser.add_argument("--epochs", default=10, type=int, help="The number of the epochs.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="The max value of the gradient normalization.")
    parser.add_argument("--bert_model", default='base', type=str, help="The pretrained model.")
    parser.add_argument("--use_entity_type", action="store_true", help="Whether use entity_type or not.")
    parser.add_argument("--use_entity_ids", action="store_true", help="Whether use entity_ids or not.")
    parser.add_argument("--entity_type_embed_size", default=768, type=int, help="The size of the entity type embedding.")
    parser.add_argument("--saved_embed", default='n', type=str, help="Use saved embedding or not.")
    parser.add_argument("--gcn_layers", default=1, type=int, help="The number of the GCN layers.")
    parser.add_argument("--focus", default='variant', type=str, help="The query element in the tuple.")
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
        train_dev_path = './datasets/pubmed/data/examples/document/ds_train_dev.txt'
        train_dev_list = read_files(args, train_dev_path)
        time_end = time.time()
        print('Time cost: ',time_end - time_start,'s')

        time_start = time.time()
        test_path = './datasets/pubmed/data/examples/document/jax_dev_test.txt'
        test_list = read_files(args, test_path)
        time_end = time.time()
        print('Time cost: ',time_end - time_start,'s')

        torch.save(train_dev_list, './saved_model/train_dev_pubmed.pt')
        torch.save(test_list, './saved_model/test_pubmed.pt')
    else:
        train_dev_list = torch.load('./saved_model/train_dev_pubmed.pt')
        test_list = torch.load('./saved_model/test_pubmed.pt')

if __name__ == '__main__':
    main()