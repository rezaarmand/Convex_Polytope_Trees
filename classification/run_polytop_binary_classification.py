#!/usr/bin/env python
"""
A command line interface to give an example how to use CPT
"""

import time
import argparse
import json
import pandas as pd
from copy import deepcopy
import torch


import numpy as np
np.set_printoptions(precision=4, linewidth=100, suppress=True)

from e2edt import DecisionTree


from utils.PolytopeLambda import PolytopeLambda
# Regularizer
from utils.SpatialRegularizer import SpatialRegularizer

# Datasets
from utils.load_LIBSVM import load_LIBSVM
from utils.ISBI_dataset import load_ISBI
from utils.fashionMNIST import load_MNIST
from utils.load_bace_HIV import load_bace_HIV
from utils.load_sider_tox21 import load_sider_tox21
# Visualization and evaluation
from utils.compute_score import compute_score
from utils.compute_auc import compute_auc
from utils.visualizeMNIST import export_tikz, visualize_decisions, visualizeSegmentation






from csv import writer


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)







if __name__ == '__main__':
  TIMESTAMP = time.strftime('%y-%m-%d_%H-%M-%S')

  # --- Parsing command line arguments.
  parser = argparse.ArgumentParser(
                      description="Train an End-to-end learned Deterministic "
                                  "Decision Tree.")
  parser.add_argument("--data", default='MNIST', type=str, 
                      choices=['MNIST','FashionMNIST','protein','SensIT',
                               'connect4', 'bace', 'HIV', 'sider', 'tox21'],#,'ISBI'],
                      help="Known data sets you can use.")
  parser.add_argument("--leaves", default=0, type=int,
                      help="Max number of leaves for best first training.")
  parser.add_argument("--depths", default=[2], type=int, nargs='+',
                      help="Max depth of tree. If given a list of increasing "
                           "depths, the script will fit a tree for a depth, "
                           "test it, refine it, test it again and do the same "
                           "for the next max depth starting from the current "
                           "non-refined tree.")

  parser.add_argument("--seed", default=-1, type=int,
                      help="torch seed")
  parser.add_argument("-r", "--reg", default=0.0, type=float,
                      help="Regularization factor, if regularizer is given.")
  parser.add_argument("-w", "--weightdecay", default=0.0, type=float,
                      help="Optimizer weight decay (L2).")
  parser.add_argument("-s", "--steepness_inc", default=0.1, type=float,
                      help="Steepness increase per epoch.")
  parser.add_argument("-init_s", "--initial_steepness", default=1.0, type=float,
                      help="Steepness increase per epoch.")
  #parser.add_argument("-k", "--kernel", default=9, type=int,
  #                    help="Kernel size for ISBI.")

  parser.add_argument("--epochs", default=20, type=int,
                      help="Max number of epochs per fit (i.e. each split in "
                           "greedy fitting and refinement).")
  parser.add_argument("--refine_epochs", default=20, type=int,
                      help="Max number of epochs for refining.")
  parser.add_argument("--lr", default=0.001, type=float,
                      help="Optimzer learning rate.")
  parser.add_argument("--polytope_sides", default=30, type=int,
                      help="Max number of polytope sides.")
  parser.add_argument("--base_module", default='PolytopeLambda', type=str, 
                      choices=['none','PolytopeLambda'],
                      help="Optimization algorithm to use.")
  parser.add_argument("--batch_size", default=1000, type=int,
                      help="Batch used by all loaders. "
                           "0 means a single batch. "
                           "Small batches slow down refinement! "
                           "Should be positive (no checks).")
  # parser.add_argument("--lr", default=0.001, type=float,
  #                     help="learning rate.")
  parser.add_argument("--algo", default='EM', type=str, choices=['EM','alt'],
                      help="Optimization algorithm to use.")
  #parser.add_argument("-v","--visualize", action='store_true',
  #                    help="Visualize filters.")
  parser.add_argument("--testing", action='store_true',
                      help="Use test set for testing instead of validation set.")
  parser.add_argument("--refine", action='store_true',
                      help="Run refinement at each depth.")
  parser.add_argument("--cuda", action='store_true',
                      help="Use cuda for computations.")
  args = parser.parse_args()

  print("Running with following command line arguments: {}".\
        format(args))

  # --- Set up plot folder.
  argstr = [k[0]+str(v) for k,v in vars(args).items()]
  argstr = ''.join(argstr)
  PLOT_FOLDER = 'plot/{}_{}'.format(argstr,TIMESTAMP)

  use_validation = False if args.testing else True
  # --- Load data.
  if args.data == 'MNIST':
    train_set, test_set, n_features, n_classes =\
      load_MNIST(use_validation)
  elif args.data == 'FashionMNIST':
    train_set, test_set, n_features, n_classes =\
      load_MNIST(use_validation, fashion=True)
  #elif args.data == 'ISBI':
  #  train_set, test_set, n_features, n_classes =\
  #    load_ISBI(kernel_size=args.kernel)
  elif args.data in ['protein', 'connect4', 'SensIT']:
    train_set, test_set, n_features, n_classes =\
      load_LIBSVM(args.data, use_validation)

  elif args.data in ['bace', 'HIV']:
    train_set, valid_set, test_set, n_features, n_classes =\
      load_bace_HIV(args.data)
  elif args.data in ['sider', 'tox21']:
    train_set, valid_set, test_set, n_features, n_classes =\
      load_sider_tox21(args.data)

  print("Data set properties: features = {}, classes = {}".\
        format(n_features, n_classes))

  if args.seed > 0:
    torch.manual_seed(args.seed)
  else:
    args.seed = np.random.randint(100)
    torch.manual_seed(args.seed)

  # --- Model setup.

  if args.base_module == 'PolytopeLambda':
    non_linear_module = PolytopeLambda

  regularizer = None
  if args.reg > 0:
    regularizer = SpatialRegularizer(n_features, strength=args.reg, cuda=args.cuda)

  batch_size = args.batch_size
  if batch_size == 0:
    batch_size = len(train_set)

  # steepness for CNN 25, for rest: 1
  model = DecisionTree(n_features, n_classes, initial_steepness=args.initial_steepness, 
                       regularizer=regularizer,
                       non_linear_module=non_linear_module,
                       polytope_sides=args.polytope_sides,
                       batch_size=batch_size,
                       lr=args.lr
                       )
  
  if args.cuda:
    model.cuda()

  #~# --- Code to start from a randomly initialized tree.
  #~model.build_balanced(args.depths[-1], 1.0)
  #~for f_loss in model.refine(train_loader, args.epochs, algo=args.algo):
  #~  print(f_loss)
  #~  print(compute_score(model, train_loader))
  #~  print(compute_score(model, test_loader))

  steepnesses = [1.0]
  greedy_scores = []
  ref_scores = {s: [] for s in steepnesses}
  # --- Iterate over different depths. Each time, first build a greedy tree up
  # to given depth, then refine model.
  for depth in args.depths:
    print("Current max tree depth = {}".format(depth))

    # --- greedy training
    model.fit_greedy(train_set, epochs=args.epochs, 
                     max_depth=depth, algo=args.algo, 
                     steepness_inc=args.steepness_inc,
                     weight_decay=args.weightdecay,
                     n_max_nodes=args.leaves)

    #~# Nasty hack to count leaves without implementing a function.
    #~count = [0]
    #~model.root.foreach(lambda n: count.__setitem__(0,count[0] + int(n._path_end)))
    #~print("leaves = {}".format(count))

    # --- Evaluate greedy model.
    model.eval()
    train_score_discrete = compute_auc(model, train_set, cuda=args.cuda)
    valid_score_discrete = compute_auc(model, valid_set, cuda=args.cuda)
    test_score_discrete = compute_auc(model, test_set, cuda=args.cuda)
    
    greedy_scores.append([depth, train_score_discrete,
     valid_score_discrete, test_score_discrete])

    print("Greedy tree scores at depth = {}:\n"
          "\t Deterministic: Train score = {:.4}, Valid score = {:.4}, Test score = {:.4}".\
          format(greedy_scores[-1][0], greedy_scores[-1][1],
                 greedy_scores[-1][2], greedy_scores[-1][3]))

    # --- Refine current model. Not available with cuda enabled, due to
    # deepcopy.
    if args.refine:
      print("Refining")
      failed = True
      for steepness in steepnesses:
        #print("Steepness = {}".format(steepness))
        if len(args.depths) >1:
          ref_model = deepcopy(model)
        else:
          ref_model = model

        ref_epoch = 0
        for f_loss in ref_model.refine(train_set, args.refine_epochs, args.algo, 
                                       args.weightdecay):
          ref_epoch += 1
          if ref_epoch % 5 == 0:
            print("Epoch {}".format(ref_epoch))
          if np.isnan(f_loss):
            print("failed!")
            failed = True
            break
          ref_model.root.set_steepness(inc_value=args.steepness_inc)

        # --- Evaluate the refined model.
        ref_model.eval()
        train_score_discrete = compute_auc(ref_model, train_set, cuda=args.cuda)
        valid_score_discrete = compute_auc(ref_model, valid_set, cuda=args.cuda)
        test_score_discrete = compute_auc(ref_model, test_set, cuda=args.cuda)
        
        ref_scores[steepness].append([depth,
                           train_score_discrete,
                           valid_score_discrete,
                           test_score_discrete])
        print("Refined tree scores at depth = {}:\n"
              "\t Deterministic: Train score = {:.4}, Valid score = {:.4}, Test score = {:.4}".\
              format(ref_scores[steepness][-1][0],ref_scores[steepness][-1][1],
                ref_scores[steepness][-1][2],ref_scores[steepness][-1][3]))

  # --- Print summary of scores.
  print("\n--- Summary")
  print("Scores for greedy trees at specific depth.")

  ref_scores_list = [scores for _, scores in ref_scores.items()][0]
  print(ref_scores)
  print(ref_scores_list)

  # pd_greedy_scores = pd.DataFrame(greedy_scores, columns=['depth', 'Train score', 'Valid score' , 'Test score'])

  # pd_ref_scores = pd.DataFrame(ref_scores_list, columns=['depth', 'Train score', 'Valid score' , 'Test score'])

  # if pd_greedy_scores['Valid score'].max() >= pd_ref_scores['Valid score'].max():

  #   test_acc_output = pd_greedy_scores['Test score'][pd_greedy_scores['Valid score'].idxmax()]
  # else:

  #   test_acc_output = pd_ref_scores['Test score'][pd_ref_scores['Valid score'].idxmax()]
  # print(test_acc_output)
  # file_name_str = args.data+"{:.4}-epch{}-ref_ep{}-s{}init_s{}-pol_sd{}".format(
  #   test_acc_output, args.epochs, args.refine_epochs, args.steepness_inc, args.initial_steepness, args.polytope_sides)
  # pd_greedy_scores.to_csv('G_'+file_name_str+'.csv', index=False)
  # pd_ref_scores.to_csv('R_'+file_name_str+'.csv', index=False)

  pd_greedy_scores_ = pd.DataFrame(greedy_scores, columns=['depth', 'Train score', 'Valid score' , 'Test score'])

  pd_ref_scores_ = pd.DataFrame(ref_scores_list, columns=['depth', 'Train score', 'Valid score' , 'Test score'])

  for depth_ in range(len(pd_greedy_scores_)-1):


    pd_greedy_scores = pd_greedy_scores_.iloc[:depth_+2]
    pd_ref_scores = pd_ref_scores_.iloc[:depth_+2]
    if pd_greedy_scores['Valid score'].max() >= pd_ref_scores['Valid score'].max():

      test_acc_output = pd_greedy_scores['Test score'][pd_greedy_scores['Valid score'].idxmax()]
      valid_acc_output = pd_greedy_scores['Valid score'][pd_greedy_scores['Valid score'].idxmax()]
    else:

      test_acc_output = pd_ref_scores['Test score'][pd_ref_scores['Valid score'].idxmax()]
      valid_acc_output = pd_ref_scores['Valid score'][pd_ref_scores['Valid score'].idxmax()]
    print(test_acc_output)
    file_name_str = args.data+"epch{}-ref_ep{}-s{}init_s{}-pol_sd{}-batch_size{}-lr{}".format(
       args.epochs, args.refine_epochs, args.steepness_inc, args.initial_steepness, args.polytope_sides, args.batch_size, args.lr)

    append_list_as_row(args.data+'results.csv', [test_acc_output, depth_, valid_acc_output, args.seed])
    append_list_as_row(args.data+'results.csv', ["".join(file_name_str)])

  append_list_as_row(args.data+'results.csv', ['greedy', 'Valid score'] + pd_greedy_scores_['Valid score'].to_list())
  append_list_as_row(args.data+'results.csv', ['greedy', 'Test score'] + pd_greedy_scores_['Test score'].to_list())

  append_list_as_row(args.data+'results.csv', ['refined', 'Valid score']+ pd_ref_scores_['Valid score'].to_list())
  append_list_as_row(args.data+'results.csv', ['refined', 'Test score'] + pd_ref_scores_['Test score'].to_list())

  for score in greedy_scores:
    print("Depth = {}:\n"
          "\t Deterministic: Train score = {:.4}, Valid score = {:.4}, Test score = {:.4}".\
          format(score[0], score[1], score[2], score[3]))
  print("\nScores for refined trees at specific depth.")
  for s, scores in ref_scores.items():
    for score in scores:
      print("Depth = {}:\n"
          "\t Deterministic: Train score = {:.4}, Valid score = {:.4}, Test score = {:.4}".\
          format(score[0], score[1], score[2], score[3]))

  ## --- Visualize (not tested).
  #if args.visualize:
  #  visualize_decisions(model, PLOT_FOLDER)

  #if args.data == "ISBI":
  #  print("Visualize ISBI")
  #  visualizeSegmentation(model, test_set, 'plot/images/{}_{}_test'.format(TIMESTAMP, argstr), weighted=False)

