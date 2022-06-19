import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from low_level_utils import *

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=None):
        super(GraphConvolution, self).__init__()
        # self.hidden_size = hidden_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias_flag = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        init_range = 0.1
        nn.init.uniform_(self.weight, -init_range, init_range)
        if self.bias_flag:
            nn.init.uniform_(self.bias, -init_range, init_range)

    def forward(self, args, inputm, adjm):
        support = torch.matmul(inputm.to(args.device), self.weight.to(args.device))
        output = torch.matmul(adjm.to(dtype=torch.float).to(args.device), support.to(dtype=torch.float).to(args.device))
 
        if self.bias is not None:
            return output + self.bias.repeat(output.size(0), output.size(1), 1)
        else:
            return output

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) 
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, alpha, nheads):
        super(GAT, self).__init__()
        self.attentions = [GraphAttentionLayer(nfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        return x

class scirex_bert(nn.Module):
    def __init__(self, args):
        super(scirex_bert, self).__init__()
        self.args = args
        self.hidden_size = 768
        if self.args.dataset == 'scirex':
            self.num_aspects = 4
        elif self.args.dataset == 'pubmed':
            self.num_aspects = 2
        elif self.args.dataset == 'tdms':
            self.num_aspects = 3
        
        self.input_size = self.num_aspects*4
        self.num_heads = args.gat_headers
        self.GCN_layers = nn.ModuleList([GraphConvolution(+self.input_size, self.input_size, bias=None) for i in range(self.args.gcn_layers)])
        
        self.GAT_layers = nn.ModuleList([GAT(self.input_size, self.input_size, 0.2, self.num_heads) for i in range(self.args.gcn_layers)])
        
        self.classifier1 = nn.Sequential(nn.Linear((self.input_size)*self.num_heads, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 2))
        self.classifier2 = nn.Sequential(nn.Linear((self.hidden_size)*(self.num_aspects+1), 2))
        self.node_affine = nn.Linear(self.hidden_size, 256)
        self.query_affine = nn.Linear(self.hidden_size, 256)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()
        self.device = args.device
        self.softmax = nn.Softmax(-1)
        if args.partial:
            self.partial = True
        else:
            self.partial = False

    def forward(self, doc, tup_idx, train_flag):
        predictions = []
        labels = []
        logits = []
        loss = 0
        total_examples = 0
        acc = 0
        sim_graph = torch.max(torch.cat((doc.sim_graph1.unsqueeze(-1), doc.sim_graph2.unsqueeze(-1), doc.sim_graph3.unsqueeze(-1)), -1),-1).values.squeeze(-1)

        if self.args.edges == 'c':
            adjacent_graph = doc.coref_graph + sim_graph
        elif self.args.edges == 'cc':
            adjacent_graph = doc.coref_graph + doc.cooccur_graph + sim_graph + doc.align_graph
        elif self.args.edges == 'ccr':
            adjacent_graph = doc.coref_graph + doc.cooccur_graph + doc.ref_graph + sim_graph + doc.align_graph
        
        for i in range(adjacent_graph.shape[0]):
            adjacent_graph[i,i] = 1
        A = adjacent_graph
        D = torch.sum(A, -1)
        L = D - A
        D = D ** (-1/2)
        D = torch.diag(D)
        A = torch.matmul(torch.matmul(D, A), D)
        adjacent_graph = A
        if self.partial == True:
            entity_ids_set = []
            for i in range(len(doc.entity_ids_label[doc.para_labels[tup_idx]])):
                if self.args.dataset == 'scirex' or self.args.dataset == 'tdms':
                    if doc.entity_ids_label[doc.para_labels[tup_idx]][i] != -1 and doc.entity_ids_label[doc.para_labels[tup_idx]][i] not in entity_ids_set and doc.entity_types[doc.entity_ids_label[doc.para_labels[tup_idx]][i]] == 'score':
                        entity_ids_set.append(doc.entity_ids_label[doc.para_labels[tup_idx]][i])
                else:
                    if doc.entity_ids_label[doc.para_labels[tup_idx]][i] != -1 and doc.entity_ids_label[doc.para_labels[tup_idx]][i] not in entity_ids_set:
                        entity_ids_set.append(doc.entity_ids_label[doc.para_labels[tup_idx]][i])
            entity_ids_set.sort()
        else:
            entity_ids_set = range(len(doc.entities))
        
        new_embedding = doc.scores[tup_idx].to(self.device)
        initial_embedding = doc.initial_embed.to(self.device)

        elements_embedding = doc.elements_embedding[tup_idx].to(self.device)
        elements_embedding = torch.reshape(elements_embedding, (-1,))
        current_embedding = torch.cat((elements_embedding.repeat(len(initial_embedding), 1), initial_embedding), -1)
        for layer in self.GAT_layers:
            new_embedding = layer(new_embedding, adjacent_graph.to(self.device))

        new_embedding = new_embedding[entity_ids_set]
        current_embedding = current_embedding[entity_ids_set]
        score1 = self.softmax(self.classifier1(new_embedding.to(self.device)))
        score2 = self.softmax(self.classifier2(current_embedding.to(self.device)))
        score = (score1 + score2) / 2
        score = score[:,1]
        score1 = score1[:,1]
        score2 = score2[:,1]

        if self.partial:
            label = entity_ids_set.index(int(doc.tup_label[tup_idx]))
        else:
            label = int(doc.tup_label[tup_idx])

        if self.args.dataset == 'scirex' or self.args.dataset == 'tdms':
            label_list = []
            idi = 0
            for i in entity_ids_set:
                if doc.gt_tuples[tup_idx][-1] in doc.entities[i]:
                    label_list.append(idi)
                idi += 1
        elif self.args.dataset == 'pubmed':
            label_list = []
            idi = 0
            for i in entity_ids_set:
                if doc.gt_tuples[tup_idx][-1] in ''.join(doc.entities[i]):
                    label_list.append(idi)
                idi += 1
    
        loss = torch.sum(-torch.log(torch.ones_like(score)-score))
        for l in label_list:
            loss = loss + torch.log(1 - score[l]) - torch.log(score[l])

        loss1 = torch.sum(-torch.log(torch.ones_like(score1)-score1))
        for l in label_list:
            loss1 = loss1 + torch.log(1 - score1[l]) - torch.log(score1[l])

        loss2 = torch.sum(-torch.log(torch.ones_like(score2)-score2))
        for l in label_list:
            loss2 = loss2 + torch.log(1 - score2[l]) - torch.log(score2[l])

        loss3 = torch.sum((score1-score2)**2)

        loss = loss + 0.3*loss1 + 0.3*loss2 + 0.15*loss3

        if self.args.dataset == 'scirex' or self.args.dataset == 'tdms':
            if torch.argmax(score) == label or doc.entities[entity_ids_set[torch.argmax(score)]] == doc.gt_tuples[tup_idx][-1]:
                acc += 1
        elif self.args.dataset == 'pubmed':
            if torch.argmax(score) == label or ''.join(doc.entities[entity_ids_set[torch.argmax(score)]]) == doc.gt_tuples[tup_idx][-1]:
                acc += 1
        labels.append(label)
        logits.append(score)
        predictions.append(entity_ids_set[torch.argmax(score)])
        prediction = entity_ids_set[torch.argmax(score)]

        if train_flag:
            return score, label, loss, acc
        else:
            return score, label, acc, prediction

