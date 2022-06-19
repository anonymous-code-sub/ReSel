import numpy as np
import torch
import shutil
import torch.autograd as Variable
import random
import os
import Levenshtein
import csv
import re

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

def flat_accuracy_more(preds, labels):
    pred_flat = np.argmax(preds, axis=-1).flatten()
    labels_flat = labels.flatten()
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(pred_flat)):
        if (pred_flat[i] == labels_flat[i]) and (pred_flat[i] == 1):
            tp += 1
        elif (pred_flat[i] != labels_flat[i]) and (pred_flat[i] == 1):
            fp += 1
        elif (pred_flat[i] == labels_flat[i]) and (pred_flat[i] == 0):
            tn += 1
        elif (pred_flat[i] != labels_flat[i]) and (pred_flat[i] == 0):
            fn += 1
    return np.sum(pred_flat == labels_flat), tp, fp, tn, fn

def flat_accuracy_zs(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

def RemoveNums(s):
    return re.sub('[0-9.]', '', s)

def MaskNum(s):
    return re.sub('[0-9]', '1', s)

def LevenDist(x, y):
    if len(x) * len(y) == 0: return 1
    return Levenshtein.distance(x, y) / max(len(x), len(y))

def GetLCS2(u, v):
    u, v = u[-30:], v[-30:]
    lu, lv = len(u), len(v)
    f, g = [[0] * 31 for i in range(31)], [[0] * 31 for i in range(31)]
    f[0][0] = 0
    mmax = 0
    for i in range(1,lu+1):
        for j in range(1,lv+1):
            if u[i-1] == v[j-1]:
                f[i][j] = f[i-1][j-1] + 1
                if f[i][j] > mmax:
                    mmax = f[i][j]

            # f[i][j] = max(f[i-1][j], f[i][j-1], f[i-1][j-1] + (1 if u[i-1]==v[j-1] else 0))
            # if f[i-1][j] == f[i][j]: g[i][j] = 1
            # elif f[i][j-1] == f[i][j]: g[i][j] = 2
            # else: g[i][j] = 3
    return mmax
    # rr, x, y = [], lu, lv
    # while x > 0 and y > 0:
    #     gg = g[x][y]
    #     if gg == 3: rr.append(u[x-1])
    #     x -= gg & 1
    #     y -= (gg & 2) // 2
    # return ''.join(reversed(rr))

def LCSSim2(x, y):
    if len(x) * len(y) == 0: return 1
    return GetLCS2(x, y) / min(len(x), len(y))

def GetLCS(u, v):
    u, v = u[-30:], v[-30:]
    lu, lv = len(u), len(v)
    f, g = [[0] * 31 for i in range(31)], [[0] * 31 for i in range(31)]
    f[0][0] = 0
    for i in range(1,lu+1):
        for j in range(1,lv+1):
            f[i][j] = max(f[i-1][j], f[i][j-1], f[i-1][j-1] + (1 if u[i-1]==v[j-1] else 0))
            if f[i-1][j] == f[i][j]: g[i][j] = 1
            elif f[i][j-1] == f[i][j]: g[i][j] = 2
            else: g[i][j] = 3
    return f[lu][lv]
    rr, x, y = [], lu, lv
    while x > 0 and y > 0:
        gg = g[x][y]
        if gg == 3: rr.append(u[x-1])
        x -= gg & 1
        y -= (gg & 2) // 2
    return ''.join(reversed(rr))

def LCSSim(x, y):
    if len(x) * len(y) == 0: return 1
    return GetLCS(x, y) / max(len(x), len(y))

def flat_accuracy_doc(preds, labels):
    return np.sum(preds == labels)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

def text_table_analysis(labels_list, preds_list, criterion_list):
    text_acc, text_num, table_acc, table_num = 0, 0, 0, 0
    if (len(labels_list) == len(criterion_list)):
        for i in range(len(labels_list)):
            if labels_list[i] < criterion_list[i]:
                text_num += 1
                if labels_list[i] == preds_list[i]:
                    text_acc += 1
            else:
                table_num += 1
                if labels_list[i] == preds_list[i]:
                    table_acc += 1
        return 0,0 #text_acc/text_num, table_acc/table_num
    else:
        return 0, 0

def test(data_list, model, args, file_path):
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
    logits_list = []
    nb_steps = 0
    MAX_LEN = 512
    with open(file_path, "a+") as f:
        csv_write = csv.writer(f)
        csv_header = ["Answer", "Ground-Truth", "Prediction", "RightOrNot", "Another Answer?", "Table or Text"]
        csv_write.writerow(csv_header)
        for doc in data_list:
            for idx in range(len(doc.tuple_embedding)):
                criterion_list.append(doc.text_data_section_numbers)
                query = doc.tuple_embedding[idx]
                label = doc.para_labels[idx]
                temp_labels = []
                paras = doc.para_embeddings
                paras_embed = torch.zeros((len(paras), len(paras[0])))
                labels = torch.zeros(len(paras))
                labels[label] = 1
                for i in range(len(paras)):
                    paras_embed[i] = paras[i]
                    if doc.gt_tuples[idx][-1] in doc.parts[i]:
                        temp_labels.append(i)
                with torch.no_grad():
                    logits, loss = model(query.to(args.device), paras_embed.to(args.device), labels.to(args.device), label)
                logits = logits.detach().cpu().numpy()
                logits_list.append(logits)
                labels = labels.numpy()
                min_rank = np.inf
                for l in temp_labels:
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
                label = doc.para_labels[idx]
                pred = np.argmax(logits)
                if args.dataset == 'scirex':
                    while not 'score' in doc.entity_label[pred]:
                        logits[pred] = 0
                        pred = np.argmax(logits)
                
                # if torch.cuda.device_count() > 1:
                #     loss = loss.mean()
                # output_logits = output_logits.detach().to('cpu').numpy()
                # labels = labels.to('cpu').numpy()
                labels_list.append(label)
                preds_list.append(pred)
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
        logits_list = np.array(logits_list)
        new_file_path = file_path[:-4] + '_hl.npy'
        np.save(new_file_path, logits_list)
    return loss, accuracy, mrr, hit2, hit3, hit5, loose_acc, nb_steps, nb_examples, labels_list, preds_list, criterion_list

def IR_metric(scores, label):
    scores = list(scores)
    labeled_item = scores[label]
    scores.sort()
    rank = len(scores) - scores.index(labeled_item)
    return rank

if __name__ == '__main__':
    ou = OrnsteinUhlenbeckActionNoise(1)
    states = []
    for i in range(1000):
        states.append(ou.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()