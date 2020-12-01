import spike_queries

import pandas as pd
import tqdm
import pickle
import random
import itertools
import numpy as np
from typing import List
from typing import Dict, Tuple
from collections import defaultdict
import time
import json
import argparse

def load_queries(path):
    with open(path, "r") as f:
        queries = f.readlines()
        
    queries = [l.strip() for l in queries]
    queries = queries[:25]
    
    # assert queries are well-formatted
    
    for i,q in enumerate(queries):
        assert len(q.split("\t")) == 2

    # generate mapping between queries and ids

    id2query = defaultdict(list)

    for id_and_query in queries:

        query,id = id_and_query.split("\t")
        id2query[int(id)].append(query)
               
    return queries, id2query

def run_spike_queries(queries):

    queries2results = defaultdict(list) # will contain a mapping between query id to a list of dataframes containing results 

    dataset_name = "covid19"
    num_results = 10
    query_type = "syntactic"
    
    for i, q_and_id in tqdm.tqdm(enumerate(queries), total = len(queries)):
        q, id = q_and_id.split("\t")
        id = int(id)
        try:
            df = spike_queries.perform_query(q, dataset_name, num_results, query_type)
            queries2results[id].append(df)
            time.sleep(1)
        except Exception as e:
            print("Error when running query {}".format(i+1))
            print(e)
            continue
    
    # drop none
    for id, dfs in queries2results.items():
     for i,df in enumerate(dfs):
         queries2results[id][i] = df.dropna(subset=['arg1_first_index',"arg1_last_index","arg2_first_index","arg2_last_index"])
    
    return queries2results
    
    
def add_arguments(sent:str, arg1_start, arg1_end, arg2_start, arg2_end):
    
        s_lst = sent.split(" ")
        if arg1_start > arg2_start:
            arg1_start, arg2_start = arg2_start, arg1_start
            arg1_end, arg2_end = arg2_end, arg1_end
            arg1_str, arg2_str = "{{ARG2:", "<<ARG1:"
        else:
            arg1_str, arg2_str = "<<ARG1:", "{{ARG2:"
        
        s_with_args = s_lst[:arg1_start] + [arg1_str] + s_lst[arg1_start:arg1_end+1] + [">>"] + s_lst[arg1_end+1:arg2_start] + [arg2_str] + s_lst[arg2_start:arg2_end+1] + ["}}"] +s_lst[arg2_end+1:]  
        s_with_args = " ".join(s_with_args).replace("ARG1: ", "ARG1:").replace("ARG2: ", "ARG2:")
        s_with_args = s_with_args.replace(" >>", ">>").replace(" }}", "}}")
        return s_with_args
        
        
def collect_data(queries, queries2results, id2query, ids, num_pairs_per_relation = 1500):

 data = []
 
 
 for id in tqdm.tqdm(ids): #foreach relation
    
    pattern_pairs = list(itertools.product(range(len(id2query[id])), repeat=2))
    pattern_pairs_different = [(p1,p2) for (p1,p2) in pattern_pairs if p1 != p2]
    pattern_pairs_same = [(p1,p2) for (p1,p2) in pattern_pairs if p1 == p2]
    pattern_pairs_different += pattern_pairs_same
    sum_pattern = 0
    relation_examples = []
    
    for p1, p2 in pattern_pairs_different:
        df1,df2 = queries2results[id][p1], queries2results[id][p2]
        if df1.empty or df2.empty: continue
            
        sentences1, sentences2 = df1["sentence_text"].tolist(), df2["sentence_text"].tolist()

        query1, query2 = id2query[p1], id2query[p2] #df1["spike_query"].tolist()[0], df2["spike_query"].tolist()[0]
        arg1_first_start, arg1_first_end = df1["arg1_first_index"].tolist(), df1["arg1_last_index"].tolist()
        arg2_first_start, arg2_first_end = df1["arg2_first_index"].tolist(), df1["arg2_last_index"].tolist()
        arg1_second_start, arg1_second_end = df2["arg1_first_index"].tolist(), df2["arg1_last_index"].tolist()
        arg2_second_start, arg2_second_end = df2["arg2_first_index"].tolist(), df2["arg2_last_index"].tolist()
        
        
        all_pair_combinations = list(itertools.product(range(len(df1)), range(len(df2))))
        random.shuffle(all_pair_combinations)
        for combination in all_pair_combinations[:100]:
            ind1, ind2 = combination
            
            sent1, sent2 = sentences1[ind1], sentences2[ind2]
            sent1_arg1_start, sent1_arg1_end = arg1_first_start[ind1], arg1_first_end[ind1]
            sent1_arg2_start, sent1_arg2_end = arg2_first_start[ind1], arg2_first_end[ind1]
            sent2_arg1_start, sent2_arg1_end = arg1_second_start[ind2], arg1_second_end[ind2]
            sent2_arg2_start, sent2_arg2_end = arg2_second_start[ind2], arg2_second_end[ind2]
            sent1_with_args = add_arguments(sent1, sent1_arg1_start, sent1_arg1_end, sent1_arg2_start, sent1_arg2_end)
            sent2_with_args = add_arguments(sent2, sent2_arg1_start, sent2_arg1_end, sent2_arg2_start, sent2_arg2_end)
            
            sent1_lst = sent1.split(" ")
            sent2_lst = sent2.split(" ")
            
            sent1_arg1_w = sent1_lst[sent1_arg1_start:sent1_arg1_end+1]
            sent1_arg2_w = sent1_lst[sent1_arg2_start:sent1_arg2_end+1]
            sent2_arg1_w = sent2_lst[sent2_arg1_start:sent2_arg1_end+1]
            sent2_arg2_w = sent2_lst[sent2_arg2_start:sent2_arg2_end+1]
            
            d = {"first": sent1_with_args, "second": sent2, "second_with_arguments": sent2_with_args, "query_first": query1, "query_second": query2, 
                 "first_arg1": (sent1_arg1_start, sent1_arg1_end), "first_arg2": (sent1_arg2_start, sent1_arg2_end), 
                 "second_arg1": (sent2_arg1_start, sent2_arg1_end),
                 "second_arg2": (sent2_arg2_start, sent2_arg2_end), "first_arg1_words": " ".join(sent1_arg1_w),
                "first_arg2_words": " ".join(sent1_arg2_w), "second_arg1_words": " ".join(sent2_arg1_w), "second_arg2_words": " ".join(sent2_arg2_w),
                "relation_id": id}
            relation_examples.append(d)
    
    random.shuffle(relation_examples)
    relation_examples = relation_examples[:num_pairs_per_relation]
    data.extend(relation_examples)
    
 random.shuffle(data)
 return data
 
def clean_and_partition(data, ids):

    lengths = [len(d["first"].split(" ")) + len(d["second"].split(" ")) for d in data]
    data = [d for i,d in enumerate(data) if lengths[i] < 250]
    data = [d for d in data if (d["second_arg1"] != d["second_arg2"]) and (d["first_arg1"] != d["first_arg2"])]
    data = [d for d in data if d["first_arg1_words"] != [''] and d["first_arg2_words"] != ['']
       and d["second_arg1_words"] != [''] and d["second_arg2_words"] != ['']]
    data = [d for d in data if '' not in d["first_arg1_words"] and '' not in d["first_arg2_words"]
       and '' not in d["second_arg1_words"] and '' not in d["second_arg2_words"]]

    def is_number_between_brackets(sent, ind):
        lst = sent.split(" ")
        if not is_number(lst[ind]): return False
        if ind == len(lst) - 1 or ind == 0: return False
        return (lst[ind -1] == "[" and lst[ind + 1] == "]") or (lst[ind -1] == "(" and lst[ind + 1] == ")")

    def is_number(s):
        try:
            d = int(s)
            return True
        except:
            return False

    data = [d for d in data if not is_number_between_brackets(d["second"], d["second_arg1"][0]) and not
       is_number_between_brackets(d["second"], d["second_arg2"][0]) and not
        is_number_between_brackets(d["first"], d["first_arg1"][0]) and not
        is_number_between_brackets(d["first"], d["first_arg2"][0])]

    np.random.seed(0)
    ids_subset = np.random.choice(list(ids), size = int(0.2 * len(ids)), replace = False)
    data_train = [d for d in data if d["query_id"] not in ids_subset]
    data_dev = [d for d in data if d["query_id"] in ids_subset]
    all_data = data_train + data_dev
    return data_train, data_dev, all_data
    
    
def write(data, fname):
    
    with open(fname, "w") as f:
        for d in data:
            first, second = d["first"], d["second_with_arguments"]
            first_arg1 = d["first_arg1"]
            first_arg2 = d["first_arg2"]
            second_arg1 = d["second_arg1"]
            second_arg2 = d["second_arg2"]
        
            elems = [first, second, first_arg1, first_arg2, second_arg1, second_arg2]
            keys = ["first", "second", "first_arg1", "first_arg2"]
        
            f.write(json.dumps(d) + "\n")
            

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Creating dataset for syntactic variations and RE',
                                       )
    parser.add_argument('--input-path', dest='input_path', type=str,
                            default='./queries.txt',
                            help='path to set of SPIKE queries')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                            default="./",
                            help='output directory.')
    parser.add_argument('--pairs-per-rel', dest='pairs_per_rel', type=int,
                            default=1500,
                            help='How many paris to generate from each relation.')
    args = parser.parse_args()
    queries, id2query = load_queries(args.input_path)

    queries2results = run_spike_queries(queries)
    ids = set([int(l.split("\t")[-1]) for l in sorted(queries)])

    data = collect_data(queries, queries2results, id2query, ids, num_pairs_per_relation = args.pairs_per_rel)
    data_train, data_dev, all_data = clean_and_partition(data, ids)
    
    write(data, args.output_dir + "data1.txt")
    write(data_dev, args.output_dir + "data_dev1.txt")
    write(data_train, args.output_dir + "data_train1.txt")
