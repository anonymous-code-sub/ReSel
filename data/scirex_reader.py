import json
import jsonlines
import os
import numpy as np
import torch
from torch import nn
from torch import nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, dataloader, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
import tokenizations
from keras.preprocessing.sequence import pad_sequences
from itertools import product
import random
import time
import csv
import json
from scirex_utils import *
# from retriever import *
# from rl_bfs import BFS


number_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

class BERTdataloader(object):
    def __init__(self, tokenizer, BERTmodel, item, idx, args, reference_dict):
        super(BERTdataloader, self).__init__()
        
        self.id = item['doc_id']
        self.device = args.device

        # read tables
        tables, table_lens, table_captions, table_caption_indexes = self.table_caption_reader(self.id)
        self.num_tables = len(tables)
        
        origin_tables, origin_table_lens, origin_table_indexes, origin_table_rows, origin_table_columns = self.table_reader(self.id)
        tables, table_lens, table_captions, table_caption_indexes, table_rows, table_columns = self.table_caption_alignment(tables, table_lens, table_captions, table_caption_indexes, origin_tables, origin_table_lens, origin_table_indexes, origin_table_rows, origin_table_columns)

        # read reference list
        reference_list = reference_dict[self.id]
        words = item['words']
        words = [word.lower() for word in words]
        words_idx = len(words)
        if tables != []:
            words = words + tables
        self.entity_type_dict = {"Material":0, "Method":1, "Metric":2, "Task":3, "score":4}

        # Read the NER labels from the dataset
        ner_results, entity_ids, entity_types, new_num_entities = self.read_ner_labels(item, words, words_idx)

        # Record the section labels
        section_ids = self.read_section_labels(item, words, words_idx, table_lens)

        # Record the sentence labels
        sentence_ids = self.read_sentence_labels(item, words, words_idx, table_lens)
        self.numerical_value_idx = new_num_entities

        # add scores to the dataset
        ner_results, entity_ids, entity_types, num_entities = self.annotate_scores(words, ner_results, entity_ids, entity_types, new_num_entities, words_idx)
        
        # record the coreference labels
        if args.embed_style == 'sentences':
            parts = item['sentences']
            origin_ids = sentence_ids
        elif args.embed_style == 'paragraphs':
            parts = item['sections']
            origin_ids = section_ids
        self.text_data_section_numbers = len(parts)
        coreference = self.read_coreference_labels(words, item, parts, table_lens, num_entities, entity_ids, origin_ids)

        # Normalize the tokenization from both dataset and BERT
        input_ids, ner_results, coreference, entity_ids, origin_ids, words = self.token_normalization(args, tokenizer, num_entities, parts, words, ner_results, coreference, entity_ids, origin_ids, words_idx, table_lens, table_captions, table_caption_indexes, reference_list, table_rows, table_columns)
        self.input_ids = input_ids

        # record the positive n-Ary relation tuples
        self.read_tuple_labels(item, words, entity_ids, origin_ids, args, tokenizer, BERTmodel)
        
        # Padding the sentence and set up the sentence mask
        self.BERT_encoder(args, input_ids, BERTmodel)

        if len(self.entity_ids_label) != len(self.para_embeddings):
            print("Reading...", len(self.entity_ids_label), len(self.para_embeddings))

    def table_caption_reader(self, doc_id):
        doc_path = './datasets/scirex/tableCaptions0929/' + doc_id + '.json'
        if not os.path.exists(doc_path):
            print("WARNING: corresponding table {} does not exist!".format(doc_id))
            caption_list = []
            caption_index_list = []
            table_list = []
            final_table = []
            table_len_list = []
        else:
            caption_list = []
            caption_index_list = []
            table_list = []
            table_len_list = []
            with open(doc_path, 'r') as infile:
                contents = json.load(infile)
                for item in contents:
                    caption = item["caption"]
                    caption = caption[6:]
                    caption_index = caption[:caption.find(' ')]
                    if len(caption_index) > 0:
                        roman_number = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX' ,'XX']
                        number_dict = {'I':'1', 'II':'2', 'III':'3', 'IV':'4', 'V':'5', 'VI':'6', 'VII':'7', 'VIII':'8', 'IX':'9', 'X':'10', 
                                        'XI':'11', 'XII':'12', 'XIII':'13', 'XIV':'14', 'XV':'15', 'XVI':'16', 'XVII':'17', 'XVIII':'18', 'XIX':'19', 'XX':'20'}
                        
                        # print(doc_id, caption)
                        while not (caption_index[-1] >= '0' and caption_index[-1] <= '9') and (caption_index[-1] != 'I') and (caption_index[-1] != 'V') and (caption_index[-1] != 'X') and (len(caption_index) > 0):
                            caption_index = caption_index[:-1]
                        
                        if (caption_index[0] == 'I' or caption_index[0] == 'V' or caption_index[0] == 'X'):
                            caption_index = number_dict[caption_index]

                    caption = caption[caption.find(' ')+1:]
                    table = item["table"]
                    table = [cell.lower() for cell in table]
                    caption_index_list.append(caption_index)
                    caption_list.append(caption)
                    # table_list.append(table)
                    table_list.append(table)
                    table_len_list.append(len(table))
            
            order = np.argsort(caption_index_list)
            caption_list = [caption_list[i].lower() for i in order]
            caption_index_list = [caption_index_list[i] for i in order]
            table_len_list = [table_len_list[i] for i in order]
            final_table = []
            for i in order:
                final_table = final_table + table_list[i]
            # input()
        
        return final_table, table_len_list, caption_list, caption_index_list


    def table_reader(self, doc_id):
        doc_path = './datasets/scirex/paper_0831_scirex_withPDF/' + doc_id + '/unpack/csv/'
        # print(doc_id)
        if not os.path.exists(doc_path):
            print("WARNING: corresponding table does not exist!")
            table_seq = []
            table_lens = []
            table_indexes = []
            table_row = []
            table_column = []
        else:
            file_list = os.listdir(doc_path)
            table_seqs = []
            table_lens = []
            table_indexes = []
            table_columns = []
            table_rows = []
            for file in file_list:
                table_seq = []
                table_row = []
                table_column = []
                file_path = doc_path + file
                csv_reader = csv.reader(open(file_path))
                flag = True
                num = 0
                row_idx = 0
                for line in csv_reader:
                    column_idx = 0
                    if flag:
                        flag = False
                    else:
                        for cell in line:
                            if cell != '':
                                table_seq.append(cell.lower())
                                table_row.append(row_idx)
                                table_column.append(column_idx)
                                num += 1
                                column_idx += 1
                            else:
                                table_seq.append('[NONE]')
                                table_row.append(row_idx)
                                table_column.append(column_idx)
                                num += 1
                                column_idx += 1
                    row_idx += 1
                table_lens.append(num)
                table_index = file[:file.find('.')]
                pointer = 0
                temp = ''
                while table_index[pointer] in number_list:
                    temp = temp + table_index[pointer]
                    pointer += 1
                    if pointer >= len(table_index):
                        break
                if temp != '':
                    temp = str(int(temp) + 1) + table_index[pointer:]
                    table_index = temp
                table_indexes.append(table_index)
                table_seqs.append(table_seq)
                table_rows.append(table_row)
                table_columns.append(table_column)

            order = np.argsort(table_indexes)
            table_indexes = [table_indexes[i] for i in order]
            table_seqs = [table_seqs[i] for i in order]
            table_rows = [table_rows[i] for i in order]
            table_columns = [table_columns[i] for i in order]
            table_seq = []
            table_row = []
            table_column = []
            for i in range(len(table_seqs)):
                table_seq = table_seq + table_seqs[i]
                table_row = table_row + table_rows[i]
                table_column = table_column + table_columns[i]
            table_lens = [table_lens[i] for i in order]

        return table_seq, table_lens, table_indexes, table_row, table_column

    def table_caption_alignment(self, tables, table_lens, table_captions, table_caption_indexes, origin_tables, origin_table_lens, origin_table_indexes, origin_table_rows, origin_table_columns):
        if len(table_lens) == len(origin_table_lens):
            tables = origin_tables
            table_lens = origin_table_lens
            table_rows = origin_table_rows
            table_columns = origin_table_columns
        elif len(table_lens) != len(origin_table_lens):
            origin_table_captions = []
            origin_table_caption_indexes = []
            # print(origin_table_caption_indexes)
            for i in range(len(origin_table_indexes)):
                if origin_table_indexes[i][0] in number_list:
                    temp = ''
                    pointer = 0
                    while origin_table_indexes[i][pointer] in number_list:
                        temp = temp + origin_table_indexes[i][pointer]
                        pointer += 1
                        if pointer >= len(origin_table_indexes[i]):
                            break
                    
                    temp = int(temp)
                    if temp < len(table_captions):
                        origin_table_captions.append(table_captions[temp - 1])
                        origin_table_caption_indexes.append(table_caption_indexes[temp - 1])
                    else:
                        origin_table_captions.append('')
                        origin_table_caption_indexes.append('')
                elif not origin_table_indexes[i][-1] in number_list:
                    origin_table_captions.append('')
                    origin_table_caption_indexes.append(origin_table_indexes)
                else:
                    temp = origin_table_indexes[i][:-1] + str(int(origin_table_indexes[i][-1]) + 1)
                    if temp in table_caption_indexes:
                        temp_index = table_caption_indexes.index(temp)
                        origin_table_captions.append(table_captions[temp_index])
                        origin_table_caption_indexes.append(table_caption_indexes[temp_index])
                    else:
                        print(self.id, "NOT FOUND")
                        origin_table_captions.append('')
                        origin_table_caption_indexes.append('')
                    
            tables = origin_tables
            table_lens = origin_table_lens
            table_captions = origin_table_captions
            table_caption_indexes = origin_table_caption_indexes
            table_rows = origin_table_rows
            table_columns = origin_table_columns
        return tables, table_lens, table_captions, table_caption_indexes, table_rows, table_columns

    def read_ner_labels(self, item, words, words_idx):
        # Record the ner labels
        self.entities = []
        self.ner = item['ner']
        ner = item['ner']
        ner_results = ['O'] * len(words)
        self.entity_types = []
        entity_ids = [-1] * len(words)
        entity_types = [[], [], [], [], []]
        
        after_idx = len(ner)
        for idx in range(len(ner)):
            entity = words[ner[idx][0]:ner[idx][1]]
            entity_string = ' '.join(entity)
            self.entities.append(entity)
            ner_results[ner[idx][0]:ner[idx][1]] = [ner[idx][2]] * (ner[idx][1] - ner[idx][0])
            self.entity_types.append(ner[idx][2])
            entity_ids[ner[idx][0]:ner[idx][1]] = [idx] * (ner[idx][1] - ner[idx][0])
            entity_types[int(self.entity_type_dict[ner[idx][2]])].append(idx)
                    
        return ner_results, entity_ids, entity_types, after_idx#, coref_records

    def read_section_labels(self, item, words, words_idx, table_lens):
        section = item['sections']
        section_ids = [0] * len(words)
        for idx in range(len(section)):
            section_ids[section[idx][0]:section[idx][1]] = [idx] * (section[idx][1] - section[idx][0])
        
        
        idx_pointer = words_idx
        for idx in range(len(table_lens)):
            section_ids[idx_pointer:idx_pointer+table_lens[idx]] = [len(section) + idx] * (table_lens[idx])
            idx_pointer += table_lens[idx]    
        return section_ids

    def read_sentence_labels(self, item, words, words_idx, table_lens):
        sentences = item['sentences']
        self.sentences = sentences
        sentence_ids = [0] * len(words)
        for idx in range(len(sentences)):
            sentence_ids[sentences[idx][0]:sentences[idx][1]] = [idx] * (sentences[idx][1] - sentences[idx][0])

        idx_pointer = words_idx
        for idx in range(len(table_lens)):
            sentence_ids[idx_pointer:idx_pointer+table_lens[idx]] = [len(sentences) + idx] * (table_lens[idx])
            idx_pointer += table_lens[idx]
        
        return sentence_ids

    def annotate_scores(self, words, ner_results, entity_ids, entity_types, new_entities_num, words_idx):
        num_entities = new_entities_num
        for idx in range(words_idx):
            if (words[idx][0] >= '0') and (words[idx][0] <= '9') and (entity_ids[idx] == -1):
                if idx + 1 < len(words):
                    # print(idx, len(words))
                    if words[idx + 1][0] == '%':
                        temp_entity = words[idx] + words[idx + 1]
                        ner_results[idx] = 'score'
                        ner_results[idx + 1] = 'score'
                        entity_ids[idx] = num_entities
                        entity_ids[idx + 1] = num_entities
                    else:
                        temp_entity = words[idx]
                        ner_results[idx] = 'score'
                        entity_ids[idx] = num_entities
                else:
                    temp_entity = words[idx]
                    ner_results[idx] = 'score'
                    entity_ids[idx] = num_entities    
                self.entities.append(temp_entity)
                self.entity_types.append('score')
                entity_types[4].append(temp_entity)
                
                num_entities += 1
        return ner_results, entity_ids, entity_types, num_entities

    def read_coreference_labels(self, words, item, parts, table_lens, num_entities, entity_ids, origin_ids):
        coreference = [''] * len(words)
        self.entity_coreferences = [''] * len(self.entities)
        coref = item['coref']
        self.coref = coref
        coref_keys = [*coref]
        self.coref_graph = torch.zeros((num_entities, num_entities))
        self.coref_graph_para = torch.zeros((len(parts)+len(table_lens), len(parts)+len(table_lens)))
        for i in range(num_entities):
            self.coref_graph[i, i] = 1
        for key in coref_keys:
            examples = []
            examples_para = []
            for idx in coref[key]:
                coreference[idx[0]:idx[1]] = [key.lower()] * (idx[1] - idx[0])
                examples.append(entity_ids[idx[0]])
                examples_para.append(origin_ids[idx[0]])
            idx = [[i, j] for i in examples for j in examples]
            idx_para = [[i,j] for i in examples_para for j in examples_para]
            if idx != []:
                idx = np.array(idx)
                for i in range(len(idx)):
                    self.coref_graph[idx[i, 0], idx[i, 1]] = 1

                idx_para = np.array(idx_para)
                for i in range(len(idx_para)):
                    self.coref_graph_para[idx_para[i, 0], idx_para[i, 1]] = 1
            examples = np.array(examples)
            # print(examples)
            # input()
            for i in range(len(examples)):
                self.entity_coreferences[examples[i]] = key
            
        for i in range(len(self.entities)):
            for j in range(len(self.entities)):
                if ''.join(self.entities[i]) == ''.join(self.entities[j]) and self.entity_coreferences[i] != self.entity_coreferences[j]:
                    if self.entity_coreferences[i] == '':
                        self.entity_coreferences[i] = self.entity_coreferences[j]
                    else:
                        self.entity_coreferences[j] = self.entity_coreferences[i]
                    
        
        return coreference

    def token_normalization(self, args, tokenizer, num_entities, parts, words, ner_results, coreference, entity_ids, origin_ids, words_idx, table_lens, table_captions, caption_ids, reference_list, table_rows, table_columns):
        self.entity_label = []
        self.coref_label = []
        self.entity_ids_label = []
        self.section_ids_label = []
        input_ids = []
        total_num = len(parts) + len(table_captions)
        self.reference_table = torch.zeros((total_num, total_num))
        self.cooccur_graph = torch.zeros((num_entities, num_entities))
        self.align_graph = torch.zeros((num_entities, num_entities))
        for i in range(num_entities):
            self.cooccur_graph[i, i] = 1
        self.parts = []
        self.parts_bert = []
        self.text_caption = []
        idx_num = 0
        idx_ref = 0
        self.gt_num_entities = num_entities
        for idx in parts:
            sentence_origin = words[idx[0]:idx[1]]
            entity_origin = ner_results[idx[0]:idx[1]]
            coref_origin = coreference[idx[0]:idx[1]]
            entity_ids_origin = entity_ids[idx[0]:idx[1]]

            section_ids_origin = origin_ids[idx[0]:idx[1]]
            temp_sentence = sentence_origin
            
            while 'Table' in temp_sentence:
                table_indexes = temp_sentence.index('Table')
                if table_indexes + 1 < len(temp_sentence):
                    next_item = temp_sentence[table_indexes+1]
                else:
                    next_item = '-'
                if table_indexes + 2 < len(temp_sentence):
                    next_next_item = temp_sentence[table_indexes+2]
                else:
                    next_next_item = '-'
                if next_item[0] in number_list:
                    temp_index = ''
                    pointer = 0
                    while (next_item[pointer] in number_list) and (pointer < len(next_item)):
                        temp_index = temp_index + next_item[pointer]
                        pointer +=1
                        if pointer >= len(next_item):
                            break
                    temp_index = int(temp_index) - 1
                    if temp_index < len(table_captions):
                        self.reference_table[idx_num, len(parts) + temp_index] = 1
                elif next_item == '.' and next_next_item[0] in number_list:
                    temp_index = ''
                    pointer = 0
                    while (next_next_item[pointer] in number_list)  and (pointer < len(next_next_item)):
                        temp_index = temp_index + next_next_item[pointer]
                        pointer +=1
                        if pointer >= len(next_next_item):
                            break
                    temp_index = int(temp_index) - 1
                    if temp_index < len(table_captions):
                        self.reference_table[idx_num, len(parts) + temp_index] = 1
                temp_sentence = temp_sentence[table_indexes+1:]

            temp_sentence = sentence_origin
            while 'reference' in temp_sentence:
                ref_indexes = temp_sentence.index('reference')
                # print(temp_sentence, ref_indexes, idx_ref)
                if ref_indexes - 1 >= 0:
                    prev_item = temp_sentence[ref_indexes - 1]
                else:
                    prev_item = ''
                if ref_indexes + 1 < len(temp_sentence):
                    next_item = temp_sentence[ref_indexes + 1]
                else:
                    next_item = ''
                if prev_item == '[' and next_item == ']':
                    reference = reference_list[idx_ref]
                    idx_ref += 1
                    if reference[0] == 'Table':
                        current_item = reference[1]
                        temp_ref = ''
                        pointer = 0
                        while current_item[pointer] in number_list and pointer < len(current_item):
                            temp_ref = temp_ref + current_item[pointer]
                            pointer += 1
                            if pointer >= len(current_item):
                                break
                        # print(current_item, caption_ids)
                        if current_item in caption_ids:
                            ref_idx = caption_ids.index(current_item)
                            self.reference_table[idx_num, len(parts) + ref_idx] = 1
                        elif temp_ref != '':
                            if int(temp_ref) < len(table_captions):
                                self.reference_table[idx_num, len(parts) + int(temp_ref)] = 1
                        # print(temp_sentence)
                        # print('Find Table Reference!!!!!!!!!')
                        
                temp_sentence = temp_sentence[ref_indexes+1:]
                # input()

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
            # if self.id == '04957e40d47ca89d38653e97f728883c0ad26e5d':
            #     print(sentence_bert)
            #     print(entity_norm)
            #     print(entity_ids_norm)
            #     print(len(sentence_bert), len(entity_norm), len(entity_ids_norm))
            #     input()

            self.entity_label.append(entity_norm)
            self.coref_label.append(coref_norm)
            self.entity_ids_label.append(entity_ids_norm)
            self.section_ids_label.append(section_ids_norm)
            
        old_ids_norm = []
        for idx in self.sentences:    
            entity_ids_origin = entity_ids[idx[0]:idx[1]]
            entity_ids_norm = list(set(entity_ids_origin))
            idxs = [[i, j] for i in entity_ids_norm for j in entity_ids_norm if (i!=-1) and (j!=-1)]# and (i!=j) ]
            if idxs != []:
                idxs = np.array(idxs)
                self.cooccur_graph[idxs[:,0], idxs[:,1]] = 1
            old_idxs = [[i, j] for i in entity_ids_norm for j in old_ids_norm if (i!=-1) and (j!=-1)]
            if old_idxs != []:
                old_idxs = np.array(old_idxs)
                self.cooccur_graph[old_idxs[:,0], old_idxs[:,1]] = 0.5
                self.cooccur_graph[old_idxs[:,1], old_idxs[:,0]] = 0.5
            old_ids_norm = entity_ids_norm
        
        table_idx = words_idx
        
        caption_idx = 0
        rc_idx = 0
        new_ner_results = ner_results[:table_idx]
        new_coreference = coreference[:table_idx]
        new_entity_ids = entity_ids[:table_idx]
        new_origin_ids = origin_ids[:table_idx]
        new_words = words[:table_idx]
        table_entity_ids = []
        for idx in table_lens:
            # print(table_idx, idx)
            caption = table_captions[caption_idx]
            caption_idx += 1
            # caption = ' '.join(caption_origin)
            self.text_caption.append(caption)
            caption_bert = ['[CLS]'] + tokenizer.tokenize(caption) + ['[SEP]']
            # table = ' '.join(table_origin)
            if args.table_style == 'table':
                table_origin = words[table_idx:table_idx+idx]
                entity_origin = ner_results[table_idx:table_idx+idx]
                coref_origin = coreference[table_idx:table_idx+idx]
                entity_ids_origin = entity_ids[table_idx:table_idx+idx]
                section_ids_origin = origin_ids[table_idx:table_idx+idx]
                table = ' '.join(table_origin)
                table_bert = ['[CLS]'] + tokenizer.tokenize(table) + ['[SEP]']
            elif args.table_style == 'caption':
                table = caption
                table_bert = ['[CLS]'] + tokenizer.tokenize(table) + ['[SEP]']
            elif args.table_style == 'caption+table':
                table_origin = caption.split(' ') + words[table_idx:table_idx+idx]
                caption_len_origin = len(caption.split(' '))
                entity_origin = ['O'] * len(caption.split(' ')) + ner_results[table_idx:table_idx+idx]
                
                coref_origin = [''] * len(caption.split(' ')) + coreference[table_idx:table_idx+idx]
                
                entity_ids_origin = [-1] * len(caption.split(' ')) + entity_ids[table_idx:table_idx+idx]
                
                section_ids_origin = [origin_ids[table_idx]] * len(caption.split(' ')) + origin_ids[table_idx:table_idx+idx]
                
                row_ids_origin = [-1] * len(caption.split(' ')) + table_rows[rc_idx:rc_idx+idx]
                
                column_ids_origin = [-1] * len(caption.split(' ')) + table_columns[rc_idx:rc_idx+idx]
                
                rc_idx += idx

                table = caption + ' [SEP] ' + ' '.join(table_origin)
                # print(table)
                table_bert = ['[CLS]'] + tokenizer.tokenize(table) + ['[SEP]']
                table_idx += idx
            self.parts.append(' '.join(table_origin))

            # Use the bert tokenizer to tokenize the sentences
            self.parts_bert.append(table_bert)
            # with torch.no_grad():
            #     logits = table_ner_model(table_bert)
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
                    temp = table_ner_model(part.to(args.device))[0]
                    temp = temp.squeeze(0).detach().cpu().numpy()
                    logits[t_idx:t_idx+512] = temp
                    t_idx = t_idx + 512
                part = embed_tensor[t_idx:]
                if len(part) != 0:
                    part = torch.unsqueeze(part, 0)
                    temp =  table_ner_model(part.to(args.device))[0]
                    logits[t_idx:] = temp.squeeze(0).detach().cpu().numpy()
                predictions = np.argmax(logits, axis=-1)
                entity_table_norm = []
                for i in range(len(predictions)):
                    entity_table_norm.append(inverse_entity_type_dict[predictions[i]])
                # print(entity_norm)
            input_ids.append(embedding)
            if len(table_bert) != len(embedding):
                print(len(table_bert), len(embedding))
            
            # print(len(row_ids_origin), len(entity_ids_origin), len(column_ids_origin))

            # Normalization with the package tokenizations
            a2b, b2a = tokenizations.get_alignments(table_origin, table_bert)
            entity_origin_new = []
            for i in range(len(a2b)):
                if a2b[i] == []:
                    entity_origin_new.append('O')
                else:
                    entity_origin_new.append(entity_table_norm[a2b[i][0]])
            old_num_entities = num_entities
            entity_loc = []
            # for i in range(len(entity_origin)):
            # print(table_origin)
            # print(table_origin[:caption_len_origin])
            # input()
            for i in range(caption_len_origin):
                if entity_origin_new[i] == 'O' and entity_origin[i] != 'O':
                    entity_origin_new[i] = entity_origin[i]
                elif entity_origin_new[i] != 'O' and entity_origin[i] != 'O':
                    entity_origin_new[i] = entity_origin[i]
                elif entity_origin_new[i] != 'O' and entity_origin[i] == 'O':
                    self.entities.append(table_origin[i])
                    self.entity_types.append(entity_origin_new[i])
                    self.entity_coreferences.append('')
                    entity_loc.append(i)
                    entity_ids_origin[i] = num_entities
                    coref_origin[i] = ''
                    num_entities += 1

            for i in range(len(entity_origin_new) - caption_len_origin):
                if entity_origin[i] == 'O':
                    if table_origin[i+caption_len_origin][0] in number_list:
                        entity_origin_new[i+caption_len_origin] = 'score'
                        coref_origin[i+caption_len_origin] = ''
                    else:
                        entity_origin_new[i+caption_len_origin] = 'O'
                        coref_origin[i+caption_len_origin] = ''

                    self.entities.append(table_origin[i+caption_len_origin])
                    self.entity_types.append(entity_origin_new[i+caption_len_origin])
                    self.entity_coreferences.append(coref_origin[i+caption_len_origin])
                    entity_loc.append(i+caption_len_origin)
                    entity_ids_origin[i+caption_len_origin] = num_entities
                    num_entities += 1

            # print(table_origin)
            new_ner_results = new_ner_results + entity_origin_new
            new_coreference = new_coreference + coref_origin
            new_entity_ids = new_entity_ids + entity_ids_origin
            new_origin_ids = new_origin_ids + section_ids_origin
            new_words = new_words + table_origin
            table_entity_ids.append(entity_ids_origin)
            
            old_coref_graph = self.coref_graph
            self.coref_graph = torch.zeros((num_entities, num_entities))
            self.coref_graph[:old_num_entities, :old_num_entities] = old_coref_graph
            for i in range(num_entities):
                self.coref_graph[i, i] = 1

            old_cooccur_graph = self.cooccur_graph
            self.cooccur_graph = torch.zeros((num_entities, num_entities))
            self.cooccur_graph[:old_num_entities, :old_num_entities] = old_cooccur_graph
            for i in range(num_entities):
                self.cooccur_graph[i, i] = 1
            
            old_align_graph = self.align_graph
            self.align_graph = torch.zeros((num_entities, num_entities))
            self.align_graph[:old_num_entities, :old_num_entities] = old_align_graph
            for i in range(num_entities):
                self.align_graph[i, i] = 1
            
            entity_norm = []
            coref_norm = []
            entity_ids_norm = []
            row_ids_norm = []
            column_ids_norm = []
            for i in range(len(b2a)):
                if b2a[i] == []:
                    entity_norm.append('O')
                    coref_norm.append('')
                    entity_ids_norm.append(-1)
                    section_ids_norm.append(-1)
                    row_ids_norm.append(-1)
                    column_ids_norm.append(-1)
                else:
                    entity_norm.append(entity_origin_new[b2a[i][0]])
                    coref_norm.append(coref_origin[b2a[i][0]])
                    entity_ids_norm.append(entity_ids_origin[b2a[i][0]])
                    section_ids_norm.append(section_ids_origin[b2a[i][0]])
                    row_ids_norm.append(row_ids_origin[b2a[i][0]])
                    column_ids_norm.append(column_ids_origin[b2a[i][0]])
            self.entity_label.append(entity_norm)
            self.coref_label.append(coref_norm)
            self.entity_ids_label.append(entity_ids_norm)
            self.section_ids_label.append(section_ids_norm)
            # print(self.id)
            # print(table_origin)
            # for ii in range(len(entity_ids_origin)):
            #     print(table_origin[ii], row_ids_origin[ii], column_ids_origin[ii], entity_ids_origin[ii], self.entities[entity_ids_origin[ii]])
            # input()

            for ii in range(len(row_ids_origin)):
                for jj in range(len(row_ids_origin)):
                    if (((row_ids_origin[ii] == row_ids_origin[jj]) or (column_ids_origin[ii] == column_ids_origin[jj]) or (row_ids_origin[ii] == -1) or (row_ids_origin[jj] == -1)) and ((entity_ids_origin[ii] != -1) and (entity_ids_origin[jj] != -1))) and ((self.entity_types[entity_ids_origin[ii]] != 'score' and self.entity_types[entity_ids_origin[jj]] == 'score') or (self.entity_types[entity_ids_origin[ii]] == 'score' and self.entity_types[entity_ids_origin[jj]] != 'score')):
                    # if (((row_ids_origin[ii] == row_ids_origin[jj]) or (column_ids_origin[ii] == column_ids_origin[jj])) and ((entity_ids_origin[ii] != -1) and (entity_ids_origin[jj] != -1))) and ((self.entity_types[entity_ids_origin[ii]] != 'score' and self.entity_types[entity_ids_origin[jj]] == 'score') or (self.entity_types[entity_ids_origin[ii]] == 'score' and self.entity_types[entity_ids_origin[jj]] != 'score')):
                        self.align_graph[entity_ids_origin[ii], entity_ids_origin[jj]] = 1
                        # print(self.entities[entity_ids_origin[ii]], self.entities[entity_ids_origin[jj]])
                        # print(entity_ids_origin[ii], entity_ids_origin[jj])
            # input()
            
            entity_ids_norm = list(set(entity_ids_norm))
            # idxs = [[i, j] for i in entity_ids_norm for j in entity_ids_norm if (i!=-1) and (j!=-1) and (i!=j) ]
            # if idxs != []:
            #     idxs = np.array(idxs)
            #     self.cooccur_graph[idxs[:,0], idxs[:,1]] = 1
            # table_idx += int(idx)
        idx_num = 0
        idx_ref = 0
        self.ref_graph = torch.zeros_like(self.cooccur_graph)
        for idx in self.sentences:
            sentence_origin = new_words[idx[0]:idx[1]]
            entity_origin = new_ner_results[idx[0]:idx[1]]
            coref_origin = new_coreference[idx[0]:idx[1]]
            entity_ids_origin = new_entity_ids[idx[0]:idx[1]]

            section_ids_origin = new_origin_ids[idx[0]:idx[1]]
            temp_sentence = sentence_origin
            
            while 'table' in temp_sentence:
                table_indexes = temp_sentence.index('table')
                if table_indexes + 1 < len(temp_sentence):
                    next_item = temp_sentence[table_indexes+1]
                else:
                    next_item = '-'
                if table_indexes + 2 < len(temp_sentence):
                    next_next_item = temp_sentence[table_indexes+2]
                else:
                    next_next_item = '-'
                if next_item[0] in number_list:
                    temp_index = ''
                    pointer = 0
                    while (next_item[pointer] in number_list) and (pointer < len(next_item)):
                        temp_index = temp_index + next_item[pointer]
                        pointer +=1
                        if pointer >= len(next_item):
                            break
                    temp_index = int(temp_index) - 1
                    if temp_index < len(table_captions):
                        entity_ids_origin2 = table_entity_ids[temp_index]
                        entity_ids_temp = list(set(entity_ids_origin))
                        entity_ids_temp2 = list(set(entity_ids_origin2))
                        for i in range(len(entity_ids_temp)):
                            for j in range(len(entity_ids_temp2)):
                                if (entity_ids_temp[i] != -1) and (entity_ids_temp2[j] != -1):
                                    self.ref_graph[entity_ids_temp[i], entity_ids_temp2[j]] = 1
                                    self.ref_graph[entity_ids_temp2[j], entity_ids_temp[i]] = 1

                elif next_item == '.' and next_next_item[0] in number_list:
                    temp_index = ''
                    pointer = 0
                    while (next_next_item[pointer] in number_list)  and (pointer < len(next_next_item)):
                        temp_index = temp_index + next_next_item[pointer]
                        pointer +=1
                        if pointer >= len(next_next_item):
                            break
                    temp_index = int(temp_index) - 1
                    if temp_index < len(table_captions):
                        entity_ids_origin2 = table_entity_ids[temp_index]
                        entity_ids_temp = list(set(entity_ids_origin))
                        entity_ids_temp2 = list(set(entity_ids_origin2))
                        for i in range(len(entity_ids_temp)):
                            for j in range(len(entity_ids_temp2)):
                                if (entity_ids_temp[i] != -1) and (entity_ids_temp2[j] != -1):
                                    self.ref_graph[entity_ids_temp[i], entity_ids_temp2[j]] = 1
                                    self.ref_graph[entity_ids_temp2[j], entity_ids_temp[i]] = 1
                temp_sentence = temp_sentence[table_indexes+1:]

            temp_sentence = sentence_origin
            while 'reference' in temp_sentence:
                ref_indexes = temp_sentence.index('reference')
                # print(temp_sentence, ref_indexes, idx_ref)
                if ref_indexes - 1 >= 0:
                    prev_item = temp_sentence[ref_indexes - 1]
                else:
                    prev_item = ''
                if ref_indexes + 1 < len(temp_sentence):
                    next_item = temp_sentence[ref_indexes + 1]
                else:
                    next_item = ''
                if prev_item == '[' and next_item == ']':
                    reference = reference_list[idx_ref]
                    idx_ref += 1
                    if reference[0] == 'table':
                        current_item = reference[1]
                        temp_ref = ''
                        pointer = 0
                        while current_item[pointer] in number_list and pointer < len(current_item):
                            temp_ref = temp_ref + current_item[pointer]
                            pointer += 1
                            if pointer >= len(current_item):
                                break
                        # print(current_item, caption_ids)
                        if current_item in caption_ids:
                            ref_idx = caption_ids.index(current_item)
                            entity_ids_origin2 = table_entity_ids[ref_idx]
                            entity_ids_temp = list(set(entity_ids_origin))
                            entity_ids_temp2 = list(set(entity_ids_origin2))
                            for i in range(len(entity_ids_temp)):
                                for j in range(len(entity_ids_temp2)):
                                    if (entity_ids_temp[i] != -1) and (entity_ids_temp2[j] != -1):
                                        self.ref_graph[entity_ids_temp[i], entity_ids_temp2[j]] = 1
                                        self.ref_graph[entity_ids_temp2[j], entity_ids_temp[i]] = 1
                        elif temp_ref != '':
                            if int(temp_ref) < len(table_captions):
                                entity_ids_origin2 = table_entity_ids[int(temp_ref)]
                                entity_ids_temp = list(set(entity_ids_origin))
                                entity_ids_temp2 = list(set(entity_ids_origin2))
                                for i in range(len(entity_ids_temp)):
                                    for j in range(len(entity_ids_temp2)):
                                        if (entity_ids_temp[i] != -1) and (entity_ids_temp2[j] != -1):
                                            self.ref_graph[entity_ids_temp[i], entity_ids_temp2[j]] = 1
                                            self.ref_graph[entity_ids_temp2[j], entity_ids_temp[i]] = 1
                        # print(temp_sentence)
                        # print('Find Table Reference!!!!!!!!!')
                        print(self.id)
                        print(temp_sentence)
                        print(next_item, next_next_item)
                        list1 = []
                        for iii in range(len(entity_ids_temp)):
                            list1.append(self.entities[entity_ids_temp[iii]])
                        list2 = []
                        for iii in range(len(entity_ids_temp2)):
                            list2.append(self.entities[entity_ids_temp2[iii]])
                        print(list1, list2, table_captions[int(temp_ref)])
                        input()
                temp_sentence = temp_sentence[ref_indexes+1:]
            idx_num += 1
        self.entity_ids = entity_ids
        return input_ids, new_ner_results, new_coreference, new_entity_ids, new_origin_ids, new_words

    def read_tuple_labels(self, item, words, entity_ids, origin_ids, args, tokenizer, BERTmodel):
        relations = item['n_ary_relations']
        methods = item['method_subrelations']
        self.gt_tuples = []
        self.questions = []
        self.tuple_embedding = []
        self.elements_embedding = []
        # self.question_embedding = []
        self.tup_num = 0
        self.tup_label = []
        self.tup_candidate = []
        self.para_labels = []
        tup_len = []
        self.question_ids = []
        content = words
        # print(len(entity_ids), len(origin_ids))
        # input()
        with torch.no_grad():
            for rel in relations:
                if args.focus == 'score':
                    tup = [rel["Material"].lower(), rel["Method"].lower(), rel["Metric"].lower(), rel["Task"].lower(), rel["score"]]
                    question = "what is the " + rel["Metric"].lower() + " score of " + rel["Method"].lower() + " on " + rel["Material"].lower() + " for the " + rel["Task"].lower() + " problem?"
                    tup_idx = [0, 1, 2, 3, 4]
                    question_idx = np.ones(len(question.split(" "))) * (-1)
                    question_idx[3] = 0
                    question_idx[6] = 1
                    question_idx[8] = 2
                    question_idx[11] = 3
                elif args.focus == 'metric':
                    tup = [rel["Material"].lower(), rel["Task"].lower(), rel["Method"].lower(), rel["Metric"].lower()]
                    question = "what is the metric that " + rel["Method"].lower() + " reports on " + rel["Material"].lower() + " for the " + rel["Task"].lower() + " Problem?"
                    tup_idx = [0, 1, 2, 3]
                    question_idx = np.ones(len(question.split(" "))) * (-1)
                    question_idx[5] = 0
                    question_idx[8] = 1
                    question_idx[11] = 2
                elif args.focus == 'method':
                    tup = [rel["Material"].lower(), rel["Task"].lower(), rel["Method"]].lower()
                    question = "what is the method proposed to solve " + rel["Task"].lower() + " on " + rel["Material"].lower() + "?"
                    tup_idx = [0, 1, 2]
                    question_idx = np.ones(len(question.split(" "))) * (-1)
                    question_idx[7] = 0
                    question_idx[9] = 1
                if (tup[-1] in content) and (tup[-1] != '-'):
                    self.gt_tuples.append(tup)
                    self.questions.append(question)
                    self.tup_num += 1
                    # print(len(words), len(entity_ids))
                    # print(tup[-1], self.entities[entity_ids[content.index(tup[-1])]])
                    if entity_ids[content.index(tup[-1])] == -1:
                        print(self.parts[origin_ids[content.index(tup[-1])]], tup[-1])
                        input()
                    self.tup_label.append(entity_ids[content.index(tup[-1])])
                    candidate_section = origin_ids[content.index(tup[-1])]
                    self.para_labels.append(candidate_section)
                    section_range = [entity_ids[i] for i in range(len(origin_ids)) if (origin_ids[i] == candidate_section) and (entity_ids[i] >= self.numerical_value_idx)]
                    temp_range = [entity_ids[i] for i in range(len(origin_ids)) if (origin_ids[i] == candidate_section)]
                    if args.partial == True:
                        if section_range == []:
                            print(max(temp_range), self.numerical_value_idx)
                            print(self.parts[candidate_section])
                            print(temp_range)
                            input()
                        self.tup_candidate.append([min(section_range), max(section_range)])
                    else:
                        self.tup_candidate.append([0, len(self.entities) - 1])
                    tup_len.append(max(section_range) - min(section_range))
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
                                elements_embedding[int(question_idx_norm[i])] += tup_embed[i].cpu()
                                num[int(question_idx_norm[i])] += 1
                        for i in range(len(elements_embedding)):
                            elements_embedding[i] = elements_embedding[i] / (num[i] * torch.ones_like(elements_embedding[i]))
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
                            elements_embedding[i] = elements_embedding[i] / (num[i] * torch.ones_like(elements_embedding[i]))
                        self.elements_embedding.append(elements_embedding)
                    torch.cuda.empty_cache()
                    if args.duplicate == 'n':
                        break
    
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
                    if self.entity_ids_label[i][j] != -1:
                        self.initial_embed[self.entity_ids_label[i][j]] += embeddings[i][j]
                        num[self.entity_ids_label[i][j]] += 1

            for i in range(len(self.entities)):
                if num[i] == 0:
                    num[i] = 1
                    self.initial_embed[i] = torch.ones(768)
            # # self.initial_embed[len(self.entities)] = torch.ones(768)
            
            for i in range(self.initial_embed.shape[0]):
                self.initial_embed[i] = self.initial_embed[i] / (num[i] * torch.ones_like(self.initial_embed[i]))
            # print(self.initial_embed)
            # input()
        #     torch.save(self.initial_embed, "./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "rl_new_nt_embed.pt")
        #     torch.save(self.para_embeddings, "./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "rl_new_" + args.embed_style + "_" + args.table_style + "_nt_embed.pt")
        #     torch.save(self.para_entity_embeddings, "./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "rl_new_" + args.embed_style + "_" + args.table_style + "_entity_nt_embed.pt")
        # else:
        #     self.initial_embed = torch.load("./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "rl_new_nt_embed.pt")
        #     self.para_embeddings = torch.load("./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "rl_new_" + args.embed_style + "_" + args.table_style + "_nt_embed.pt")
        #     self.para_entity_embeddings = torch.load("./saved_model/saved_embed/" + args.bert_model + '_' + self.id + "rl_new_" + args.embed_style + "_" + args.table_style + "_entity_nt_embed.pt")

    def compute_soft_edges(self, args):
        if args.model == 'hde' or args.model == 'gcn' or args.model == 'bongcn':
            self.sim_graph1 = torch.zeros_like(self.coref_graph)
            self.sim_graph2 = torch.zeros_like(self.coref_graph)
            self.sim_graph3 = torch.zeros_like(self.coref_graph)
            for j in range(len(self.entities) - self.gt_num_entities):
                max_sim1 = 0
                max_sim2 = 0
                max_sim3 = 0
                
                sim_list1 = []
                sim_list2 = []
                sim_list3 = []
                for i in range(self.gt_num_entities):
                    if (self.entity_types[i] != "score" and self.entity_types[self.gt_num_entities+j] != "score"):
                        sim1 = LCSSim(''.join(self.entities[i]), self.entities[self.gt_num_entities+j])
                        sim2 = 1 - LevenDist(''.join(self.entities[i]), self.entities[self.gt_num_entities+j])
                        sim3 = LCSSim2(''.join(self.entities[i]), self.entities[self.gt_num_entities+j])
                        sim_list1.append([i, sim1])
                        sim_list2.append([i, sim2])
                        sim_list3.append([i, sim3])
                        if sim1 > max_sim1:
                            max_sim1 = sim1
                        if sim2 > max_sim2:
                            max_sim2 = sim2
                        if sim3 > max_sim3:
                            max_sim3 = sim3
                for k in range(len(sim_list1)):
                    if sim_list1[k][1] == max_sim1:
                        self.sim_graph1[self.gt_num_entities+j, sim_list1[k][0]] = max_sim1
                        self.sim_graph1[sim_list1[k][0], self.gt_num_entities+j] = max_sim1
                    if sim_list2[k][1] == max_sim2:
                        self.sim_graph2[self.gt_num_entities+j, sim_list2[k][0]] = max_sim2
                        self.sim_graph2[sim_list2[k][0], self.gt_num_entities+j] = max_sim2
                    if sim_list3[k][1] == max_sim3:
                        self.sim_graph3[self.gt_num_entities+j, sim_list3[k][0]] = max_sim3
                        self.sim_graph3[sim_list3[k][0], self.gt_num_entities+j] = max_sim3
        else:
            self.sim_graph = torch.zeros_like(self.coref_graph)
            for j in range(len(self.entities) - self.gt_num_entities):
                max_sim = 0
                sim_list = []
                for i in range(self.gt_num_entities):
                    if (self.entity_types[i] != "score" and self.entity_types[self.gt_num_entities+j] != "score"):
                        sim1 = LCSSim(''.join(self.entities[i]), self.entities[self.gt_num_entities+j])
                        sim2 = 1 - LevenDist(''.join(self.entities[i]), self.entities[self.gt_num_entities+j])
                        sim3 = LCSSim2(''.join(self.entities[i]), self.entities[self.gt_num_entities+j])
                        # if sim2 > sim1:
                            # self.coref_graph[i, old_num_entities+j] = sim2
                        # else:
                        sim = max(sim1, sim2, sim3)
                        sim_list.append([i,sim])
                        if sim > max_sim:
                            max_sim = sim
                for k in range(len(sim_list)):
                    if sim_list[k][1] == max_sim:
                        self.sim_graph[self.gt_num_entities+j, sim_list[k][0]] = max_sim
                        self.sim_graph[sim_list[k][0], self.gt_num_entities+j] = max_sim

    def compute_sim_scores(self, args):
        if args.evaluation == '1' or args.evaluation == '2' or args.evaluation == '3' or args.evaluation == '4':
            self.input_size = 4
        elif args.evaluation == '5':
            self.input_size = 16
        elif args.evaluation == '6' or args.evaluation == '7' or args.evaluation == '8' or args.evaluation == '9':
            self.input_size = 3
        elif args.evaluation == '10':
            self.input_size = 12
        elif args.evaluation == '11' or args.evaluation == '12' or args.evaluation == '13' or args.evaluation == '14':
            self.input_size = 2
        elif args.evaluation == '15':
            self.input_size = 8
        self.scores = []
        for tup_idx in range(len(self.gt_tuples)):
            sim_graph = torch.max(torch.cat((self.sim_graph1.unsqueeze(-1), self.sim_graph2.unsqueeze(-1), self.sim_graph3.unsqueeze(-1)), -1),-1).values.squeeze(-1)
            # sim_graph = self.sim_graph1
        
            # adjacent_graph = torch.max(torch.cat((self.coref_graph.unsqueeze(-1), self.cooccur_graph.unsqueeze(-1), self.ref_graph.unsqueeze(-1)), -1),-1).values.squeeze(-1)
            # adjacent_graph = torch.max(torch.cat((self.coref_graph.unsqueeze(-1), self.cooccur_graph.unsqueeze(-1)), -1),-1).values.squeeze(-1)
            if args.edges == 'c':
                adjacent_graph = self.coref_graph + sim_graph
            elif args.edges == 'cc':
                adjacent_graph = self.coref_graph + self.cooccur_graph + sim_graph + self.align_graph
            elif args.edges == 'ccr':
                adjacent_graph = self.coref_graph + self.cooccur_graph + self.ref_graph + sim_graph + self.align_graph
            
            entity_ids_set = []
            for i in range(len(self.entity_ids_label[self.para_labels[tup_idx]])):
                if self.entity_ids_label[self.para_labels[tup_idx]][i] != -1 and self.entity_ids_label[self.para_labels[tup_idx]][i] not in entity_ids_set and self.entity_types[self.entity_ids_label[self.para_labels[tup_idx]][i]] == 'score':
                    entity_ids_set.append(self.entity_ids_label[self.para_labels[tup_idx]][i])
            entity_ids_set.sort()

            if args.evaluation == '6' or args.evaluation == '7' or args.evaluation == '8' or args.evaluation == '9' or args.evaluation == '10':
                elements_strings = self.gt_tuples[tup_idx][:-2]
            elif args.evaluation == '1' or args.evaluation == '2' or args.evaluation == '3' or args.evaluation == '4' or args.evaluation == '5':
                elements_strings = self.gt_tuples[tup_idx][:-1]
            elif args.evaluation == '11' or args.evaluation == '12' or args.evaluation == '13' or args.evaluation == '14' or args.evaluation == '15':
                elements_strings = self.gt_tuples[tup_idx][1:3]
            

            current_id = 0
            if args.model == 'boc':
                iterations = entity_ids_set
                scores = torch.zeros((len(entity_ids_set), self.input_size))
            elif args.model == 'bongcn':
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
                            if args.evaluation == '1' or args.evaluation == '5' or args.evaluation == '6' or args.evaluation == '10' or args.evaluation == '11' or args.evaluation == '15':
                                current_sim = 1 - LevenDist(''.join(self.entities[j]), elements_strings[k].lower())
                                if current_sim > scores[current_id, idx_k]:
                                    scores[current_id,idx_k] = current_sim
                                    # best_neighbour[k] = self.entities[j]
                                idx_k += 1
                            if args.evaluation == '2' or args.evaluation == '5' or args.evaluation == '7' or args.evaluation == '10' or args.evaluation == '12' or args.evaluation == '15':
                                current_sim = LCSSim(''.join(self.entities[j]), elements_strings[k].lower())
                                if current_sim > scores[current_id, idx_k]:
                                    scores[current_id,idx_k] = current_sim
                                    # best_neighbour[k] = self.entities[j]
                                idx_k += 1
                            if args.evaluation == '3' or args.evaluation == '5' or args.evaluation == '8' or args.evaluation == '10' or args.evaluation == '13' or args.evaluation == '15':
                                current_sim = LCSSim2(''.join(self.entities[j]), elements_strings[k].lower())
                                if current_sim > scores[current_id, idx_k]:
                                    scores[current_id,idx_k] = current_sim
                                idx_k += 1
                            if args.evaluation == '4' or args.evaluation == '5' or args.evaluation == '9' or args.evaluation == '10' or args.evaluation == '14' or args.evaluation == '15':
                                current_sim = F.cosine_similarity(self.initial_embed[j].unsqueeze(0), self.elements_embedding[tup_idx][k].unsqueeze(0), dim=-1)
                                if current_sim > scores[current_id, idx_k]:
                                    scores[current_id,idx_k] = current_sim
                                idx_k += 1
                current_id += 1
            self.scores.append(scores)
    
    def compute_sim_scores2(self, args):
        self.input_size = 16
        self.scores = []
        for tup_idx in range(len(self.gt_tuples)):
            elements_strings = self.gt_tuples[tup_idx][:-1]
            current_id = 0
            
            iterations = range(len(self.entities))
            scores = torch.zeros((len(self.entities), self.input_size))
            for i in iterations:
                idx_k = 0
                for k in range(len(elements_strings)):
                    if args.evaluation == '1' or args.evaluation == '5' or args.evaluation == '6' or args.evaluation == '10' or args.evaluation == '11' or args.evaluation == '15':
                        current_sim = 1 - LevenDist(''.join(self.entities[current_id]), elements_strings[k].lower())
                        scores[current_id,idx_k] = current_sim
                        idx_k += 1
                for k in range(len(elements_strings)):
                    if args.evaluation == '2' or args.evaluation == '5' or args.evaluation == '7' or args.evaluation == '10' or args.evaluation == '12' or args.evaluation == '15':
                        current_sim = LCSSim(''.join(self.entities[current_id]), elements_strings[k].lower())
                        scores[current_id,idx_k] = current_sim
                        idx_k += 1
                for k in range(len(elements_strings)):
                    if args.evaluation == '3' or args.evaluation == '5' or args.evaluation == '8' or args.evaluation == '10' or args.evaluation == '13' or args.evaluation == '15':
                        current_sim = LCSSim2(''.join(self.entities[current_id]), elements_strings[k].lower())
                        scores[current_id,idx_k] = current_sim
                        idx_k += 1
                for k in range(len(elements_strings)):
                    if args.evaluation == '4' or args.evaluation == '5' or args.evaluation == '9' or args.evaluation == '10' or args.evaluation == '14' or args.evaluation == '15':
                        current_sim = F.cosine_similarity(self.initial_embed[current_id].unsqueeze(0), self.elements_embedding[tup_idx][k].unsqueeze(0), dim=-1)
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
    with open(file_path, 'r') as infile:
        for item in jsonlines.Reader(infile):
            document_single = BERTdataloader(tokenizer, BERTmodel, item, idx, args, reference_dict)
            if document_single.tup_num !=0:
                document_list.append(document_single)
            else:
                invalid_doc += 1
            idx += 1
            valid_tuples += document_single.tup_num
            num_tuples += len(document_single.gt_tuples)
            for label in document_single.para_labels:
                if label < document_single.text_data_section_numbers:
                    text_ans_number += 1
                else:
                    table_ans_number += 1
            if args.softedges:
                document_single.compute_soft_edges(args)

            if args.bfs:
                path_list = []
                action_list = []
                for query_idx in range(len(document_single.para_labels)):
                    path, action = BFS(document_single, document_single.para_labels[query_idx], document_single.gt_tuples[query_idx])
                    path_list.append(path)
                    action_list.append(action)
                document_single.compute_gt_paths(path_list, action_list)
            
            if args.model == 'boc' or args.model == 'bongcn':
                document_single.compute_sim_scores2(args)

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
    parser.add_argument("--model", default='bert-base', type=str, help="The pretrained model.")
    parser.add_argument("--use_entity_type", action="store_true", help="Whether use entity_type or not.")
    parser.add_argument("--use_entity_ids", action="store_true", help="Whether use entity_ids or not.")
    parser.add_argument("--entity_type_embed_size", default=768, type=int, help="The size of the entity type embedding.")
    parser.add_argument("--saved_embed", default='n', type=str, help="Use saved embedding or not.")
    parser.add_argument("--gcn_layers", default=1, type=int, help="The number of the GCN layers.")

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    time_start = time.time()
    train_path = './datasets/scirex/test.jsonl'
    train_list = read_files(args, train_path)
    time_end = time.time()
    print('Time cost: ',time_end - time_start,'s')

if __name__ == '__main__':
    main()