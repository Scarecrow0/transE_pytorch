
import os
import json


import argparse
import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from tensorboardX import SummaryWriter

import numpy as np
import time
import datetime
import random

from utils import *
from data import *
from evaluation import *
import models


USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor

else:
    LongTensor = torch.LongTensor
    FloatTensor = torch.FloatTensor



class Config(object):
    """
        The meaning of parameters:
        self.dataset: Which dataset is used to train the model? Such as 'FB15k', 'WN18', etc.
        self.learning_rate: Initial learning rate (lr) of the model.
        self.early_stopping_round: How many times will lr decrease? If set to 0, it remains constant.
        self.L1_flag: If set to True, use L1 distance as dissimilarity; else, use L2.
        self.embedding_size: The embedding size of entities and relations.
        self.num_batches: How many batches to train in one epoch?
        self.train_times: The maximum number of epochs for training.
        self.margin: The margin set for MarginLoss.
        self.filter: Whether to check a generated negative sample is false negative.
        self.momentum: The momentum of the optimizer.
        self.optimizer: Which optimizer to use? Such as SGD, Adam, etc.
        self.loss_function: Which loss function to use? Typically, we use margin loss.
        self.entity_total: The number of different entities.
        self.relation_total: The number of different relations.
        self.batch_size: How many instances is contained in one batch?
    """
    def __init__(self):
        self.dataset = None
        self.learning_rate = 0.001
        self.early_stopping_round = 0
        self.L1_flag = True
        self.embedding_size = 100
        self.num_batches = 100
        self.train_times = 1000
        self.margin = 1.0
        self.filter = True
        self.momentum = 0.9

        self.optimizer = None
        self.loss_function = None
        self.entity_total = 0
        self.relation_total = 0
        self.batch_size = 0

    def __str__(self):
        return json.dumps(self.__dict__, indent=3)


optim_dict = {
    0: "SGD",
    1: "Adam",
    2: "Adagrad",
}

loss_dict = {
    0: "marginLoss",
}

def init_cfg(args, trainTotal):

    config = Config()
    config.dataset = args.dataset
    config.learning_rate = args.learning_rate

    config.early_stopping_round = args.early_stopping_round

    if args.L1_flag == 1:
        config.L1_flag = True
    else:
        config.L1_flag = False

    config.embedding_size = args.embedding_size
    config.batch_size = trainTotal // config.num_batches
    config.num_batches = args.num_batches
    config.train_times = args.train_times
    config.margin = args.margin

    if args.filter == 1:
        config.filter = True
    else:
        config.filter = False

    config.momentum = args.momentum

    config.optimizer = optim_dict[args.optimizer]
    config.loss_function = loss_dict[args.loss_type]

    config.entity_total = getAnythingTotal('datasets/' + config.dataset, 'entity2id.txt')
    config.relation_total = getAnythingTotal('datasets/' + config.dataset, 'relation2id.txt')
    
    return config


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    

    argparser.add_argument('-d', '--dataset', type=str)
    argparser.add_argument('-l', '--learning_rate', type=float, default=0.003)
    argparser.add_argument('-es', '--early_stopping_round', type=int, default=30)
    argparser.add_argument('-L', '--L1_flag', type=int, default=1)
    argparser.add_argument('-em', '--embedding_size', type=int, default=100)
    argparser.add_argument('-nb', '--num_batches', type=int, default=100)
    argparser.add_argument('-n', '--train_times', type=int, default=1000)
    argparser.add_argument('-m', '--margin', type=float, default=1.0)
    argparser.add_argument('-f', '--filter', type=int, default=1)
    argparser.add_argument('-mo', '--momentum', type=float, default=0.9)
    argparser.add_argument('-s', '--seed', type=int, default=0)
    argparser.add_argument('-op', '--optimizer', type=int, default=1)
    argparser.add_argument('-lo', '--loss_type', type=int, default=0)
    argparser.add_argument('-ev', '--eval_interval', type=int, default=5)
    argparser.add_argument('-np', '--num_processes', type=int, default=16)

    args = argparser.parse_args()


    if args.seed != 0:
        torch.manual_seed(args.seed)

    trainTotal, trainList, trainDict = loadTriple('./datasets/' + args.dataset, 'train2id.txt')
    validTotal, validList, validDict = loadTriple('./datasets/' + args.dataset, 'valid2id.txt')
    tripleTotal, tripleList, tripleDict = loadTriple('./datasets/' + args.dataset, 'triple2id.txt')

    config = init_cfg(args, trainTotal)

    print(str(config))

    optim_dict = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "Adagrad": optim.Adagrad,
    }

    loss_dict = {
        "marginLoss": models.MarginLoss,
    }

    loss_function = loss_dict[config.loss_function]()
    model = models.TransEModel(config)

    if USE_CUDA:
        model.cuda()
        loss_function.cuda()

    optimizer = optim_dict[config.optimizer](model.parameters(), lr=config.learning_rate)
    margin = FloatTensor([config.margin])

    start_time = time.time()

    filename = '_'.join(
        ['l', str(args.learning_rate),
         'es', str(args.early_stopping_round),
         'L', str(args.L1_flag),
         'em', str(args.embedding_size),
         'nb', str(args.num_batches),
         'n', str(args.train_times),
         'm', str(args.margin),
         'f', str(args.filter),
         'mo', str(args.momentum),
         's', str(args.seed),
         'op', str(args.optimizer),
         'lo', str(args.loss_type), ]) + '_TransE.ckpt'

    # ckpt saving dirs
    if not os.path.exists(os.path.join('model', args.dataset)):
        os.makedirs(os.path.join('model', args.dataset))

    train_batch_list = init_batch_List(trainList, config.num_batches)


    tf_writer = SummaryWriter(log_dir="tfboard", flush_secs=10)

    for epoch in range(config.train_times):
        total_loss = FloatTensor([0.0])
        random.shuffle(train_batch_list)
        for batch_items in tqdm.tqdm(train_batch_list):
            if config.filter == True:
                pos_h_batch, pos_t_batch, pos_r_batch,\
                neg_h_batch, neg_t_batch, neg_r_batch = get_filtered_batch_all(batch_items,
                                                                            config.entity_total, tripleDict)
            else:
                pos_h_batch, pos_t_batch, pos_r_batch,\
                neg_h_batch, neg_t_batch, neg_r_batch = get_raw_batch_all(batch_items,
                                                                         config.entity_total)

            batch_entity_set = set(pos_h_batch + pos_t_batch + neg_h_batch + neg_t_batch)
            batch_relation_set = set(pos_r_batch + neg_r_batch)
            batch_entity_list = list(batch_entity_set)
            batch_relation_list = list(batch_relation_set)

            # move data to GPU
            pos_h_batch = LongTensor(pos_h_batch)
            pos_t_batch = LongTensor(pos_t_batch)
            pos_r_batch = LongTensor(pos_r_batch)
            neg_h_batch = LongTensor(neg_h_batch)
            neg_t_batch = LongTensor(neg_t_batch)
            neg_r_batch = LongTensor(neg_r_batch)

            model.zero_grad()

            # link prediction loss
            pos, neg = model(pos_h_batch, pos_t_batch, pos_r_batch,
                             neg_h_batch, neg_t_batch, neg_r_batch)
            if args.loss_type == 0:
                losses = loss_function(pos, neg, margin)
            else:
                losses = loss_function(pos, neg)

            # embeddings norm loss, for regularization
            ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
            rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
            losses = losses + models.NormLoss(ent_embeddings) + models.NormLoss(rel_embeddings)

            losses.backward()
            optimizer.step()
            total_loss += losses.data


        now_time = time.time()
        print(f"Epoch {epoch}/{config.train_times}")
        # print("cost time: ", f"{now_time - start_time:.1f}", "s")
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"curr lr: {curr_lr}", )

        train_batch_loss = (total_loss/config.num_batches).item()
        print("Train batch loss: %f" % (train_batch_loss))

        tf_writer.add_scalar("train_batch_loss", train_batch_loss, epoch)
        tf_writer.add_scalar("LR", curr_lr, epoch)


        if epoch % args.eval_interval == 0:
            # Evaluate on validation set for every 5 epochs
            # randomly sample a batch of triples for validation
            if config.filter == True:
                pos_h_batch, pos_t_batch, pos_r_batch,\
                neg_h_batch, neg_t_batch, neg_r_batch = getBatch_filter_random(validList,
                                                                               config.batch_size, config.entity_total, tripleDict)
            else:
                pos_h_batch, pos_t_batch, pos_r_batch,\
                neg_h_batch, neg_t_batch, neg_r_batch = getBatch_raw_random(validList,
                                                                            config.batch_size, config.entity_total)
            pos_h_batch = LongTensor(pos_h_batch)
            pos_t_batch = LongTensor(pos_t_batch)
            pos_r_batch = LongTensor(pos_r_batch)
            neg_h_batch = LongTensor(neg_h_batch)
            neg_t_batch = LongTensor(neg_t_batch)
            neg_r_batch = LongTensor(neg_r_batch)

            pos, neg = model(pos_h_batch, pos_t_batch, pos_r_batch,
                             neg_h_batch, neg_t_batch, neg_r_batch)

            # link prediction loss
            if args.loss_type == 0:
                losses = loss_function(pos, neg, margin)
            else:
                losses = loss_function(pos, neg)

            # embeddings norm loss, for regularization
            ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
            rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
            losses = losses + models.NormLoss(ent_embeddings) + models.NormLoss(rel_embeddings)
            print("Valid batch loss: %f" % (losses.item()))

            tf_writer.add_scalar("val_batch_loss", losses.item(), epoch)

            print("start eval:")
            ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()
            rel_embeddings = model.rel_embeddings.weight.data.cpu().numpy()
            L1_flag = model.L1_flag
            filter = model.filter
            hit10, now_meanrank = evaluation_transE(validList, tripleDict, ent_embeddings, rel_embeddings,
                                                        L1_flag, filter, config.batch_size, num_processes=args.num_processes)
            print()

            tf_writer.add_scalar("val_hit@10", hit10, epoch)
            tf_writer.add_scalar("val_meanrank", now_meanrank, epoch)


            if config.early_stopping_round > 0:
                # learning rate scheduling
                if epoch == 0:
                    best_epoch = 0
                    meanrank_not_decrease_time = 0
                    lr_decrease_time = 0
                    best_meanrank = now_meanrank

                if now_meanrank < best_meanrank * 0.9:
                    meanrank_not_decrease_time = 0
                    best_meanrank = now_meanrank
                    save_dir = os.path.join('model', args.dataset, filename)
                    print("save param to: ", save_dir)
                    torch.save(model, save_dir)
                else:
                    meanrank_not_decrease_time += 1
                    # If the result hasn't improved for consecutive 3 evaluations, decrease learning rate
                    if meanrank_not_decrease_time == 3:
                        print("decrease LR")
                        lr_decrease_time += 1
                        if lr_decrease_time == config.early_stopping_round:
                            break
                        else:
                            optimizer.param_groups[0]['lr'] *= 0.5
                            meanrank_not_decrease_time = 0

            elif (epoch) % 30 == 0 or epoch == 0:
                save_dir = os.path.join('model', args.dataset, filename)
                print("save param to: ", save_dir)
                torch.save(model, save_dir)

        if epoch % 50 == 0 and epoch != 0:
            # after training, do test eval
            testTotal, testList, testDict = loadTriple('datasets/' + args.dataset, 'test2id.txt')
            oneToOneTotal, oneToOneList, oneToOneDict = loadTriple('datasets/' + args.dataset, 'one_to_one_test.txt')
            oneToManyTotal, oneToManyList, oneToManyDict = loadTriple('datasets/' + args.dataset, 'one_to_many_test.txt')
            manyToOneTotal, manyToOneList, manyToOneDict = loadTriple('datasets/' + args.dataset, 'many_to_one_test.txt')
            manyToManyTotal, manyToManyList, manyToManyDict = loadTriple('datasets/' + args.dataset, 'many_to_many_test.txt')

            ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()
            rel_embeddings = model.rel_embeddings.weight.data.cpu().numpy()
            L1_flag = model.L1_flag
            filter = model.filter

            hit10Test, meanrankTest = evaluation_transE(
                testList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=0)

            hit10OneToOneHead, meanrankOneToOneHead = evaluation_transE(
                oneToOneList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=1)
            hit10OneToManyHead, meanrankOneToManyHead = evaluation_transE(
                oneToManyList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=1)
            hit10ManyToOneHead, meanrankManyToOneHead = evaluation_transE(
                manyToOneList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=1)
            hit10ManyToManyHead, meanrankManyToManyHead = evaluation_transE(
                manyToManyList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=1)

            hit10OneToOneTail, meanrankOneToOneTail = evaluation_transE(
                oneToOneList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=2)
            hit10OneToManyTail, meanrankOneToManyTail = evaluation_transE(
                oneToManyList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=2)
            hit10ManyToOneTail, meanrankManyToOneTail = evaluation_transE(
                manyToOneList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=2)
            hit10ManyToManyTail, meanrankManyToManyTail = evaluation_transE(
                manyToManyList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=2)

            writeList = [filename, f"epohc {epoch}", "\n",
                        "eval type", "hit@10", "mean_rank", "\n",
                        'avg', '%.6f' % hit10Test, '%.6f' % meanrankTest, "\n",
                        'one_to_one_head', '%.6f' % hit10OneToOneHead, '%.6f' % meanrankOneToOneHead, "\n",
                        'one_to_many_head', '%.6f' % hit10OneToManyHead, '%.6f' % meanrankOneToManyHead, "\n",
                        'many_to_one_head', '%.6f' % hit10ManyToOneHead, '%.6f' % meanrankManyToOneHead, "\n",
                        'many_to_many_head', '%.6f' % hit10ManyToManyHead, '%.6f' % meanrankManyToManyHead, "\n",
                        'one_to_one_tail', '%.6f' % hit10OneToOneTail, '%.6f' % meanrankOneToOneTail, "\n",
                        'one_to_many_tail', '%.6f' % hit10OneToManyTail, '%.6f' % meanrankOneToManyTail, "\n",
                        'many_to_one_tail', '%.6f' % hit10ManyToOneTail, '%.6f' % meanrankManyToOneTail, "\n",
                        'many_to_many_tail', '%.6f' % hit10ManyToManyTail, '%.6f' % meanrankManyToManyTail, "\n",]

            print('\t'.join(writeList) + '\n')

            # Write the result into file
            if not os.path.exists(os.path.join('result', args.dataset)):
                os.makedirs(os.path.join('result', args.dataset))
            with open(os.path.join('result', args.dataset + '.txt'), 'a') as fw:
                fw.write('\t'.join(writeList) + '\n\n')
