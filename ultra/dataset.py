import os
import csv
import glob
from tqdm import tqdm
import copy
from functools import partial

#from ogb import linkproppred

import torch
from torch.utils import data as torch_data

from torchdrug import data, datasets, utils, core
from torchdrug.core import Registry as R


class CoDEx(torch_data.Dataset, core.Configurable):

    url = "https://github.com/tsafavi/codex/tree/master/data/triples/"
    md5 = ""
    name = "codex"

    def __init__(self, path, size=None, verbose=1):
        super(CoDEx, self).__init__()

        path = os.path.expanduser(path)
        path = os.path.join(path, f"codex-{size}")
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        
        zip_file = utils.download(self.url, path, md5=self.md5)

        print("dataset name:", self.name)

        if not os.path.exists(os.path.join(path, "train.txt")):
            train_data = utils.extract(zip_file, f"codex-{size}/train.txt")
            valid_data = utils.extract(zip_file, f"codex-{size}/valid.txt")
            test_data = utils.extract(zip_file, f"codex-{size}/test.txt")
        else:
            train_data = os.path.join(path, "train.txt")
            valid_data = os.path.join(path, "valid.txt")
            test_data = os.path.join(path, "test.txt")

        train_results = self.load_node(train_data, inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_node(valid_data, 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_node(test_data,
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])

        num_node = train_results["num_node"]
        num_relation = train_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]
        

        self.train_graph = data.Graph(train_triplets, num_node=num_node, num_relation=num_relation)
        self.valid_graph = data.Graph(valid_triplets, num_node=num_node, num_relation=num_relation)
        self.test_graph = data.Graph(test_triplets, num_node=num_node, num_relation=num_relation)
        
        self.triplets = torch.tensor(train_triplets + valid_triplets + test_triplets)
        self.graph = data.Graph(self.triplets, num_node=num_node, num_relation=num_relation)
        
        self.num_samples = [len(train_triplets), len(valid_triplets), len(test_triplets)]
        self.transform = None

    def load_node(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split()
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    def build_vocab(self):

        inv_entity_vocab, inv_rel_vocab = {}, {}
        for entity in self.codex.entities():
            if entity not in inv_entity_vocab:
                inv_entity_vocab[entity] = len(inv_entity_vocab)
        for rel in self.codex.relations():
            if rel not in inv_rel_vocab:
                inv_rel_vocab[rel] = len(inv_rel_vocab)
        
        return inv_entity_vocab, inv_rel_vocab
    
    def retrieve_text_description(self, rid):
        assert rid in self.inv_rel_vocab
        return self.codex.relation_description(rid)
    
    def load_edge(self, codex_df):
        triplets = []

        for rowidx, row in codex_df.iterrows():

            h, t, r = row['head'], row['tail'], row['relation']
            h, t = self.inv_entity_vocab[h], self.inv_entity_vocab[t]
            r = self.inv_rel_vocab[r]

            triplets.append((h, t, r))
        
        return triplets
    

    def __getitem__(self, index):
        return self.triplets[index]
        
    def __len__(self):
        return self.graph.num_edge
    
    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, list(range(offset, offset + num_sample)))
            splits.append(split)
            offset += num_sample
        return splits
    
    @property
    def num_entity(self):
        """Number of entities."""
        return self.graph.num_node

    @property
    def num_triplet(self):
        """Number of triplets."""
        return self.graph.num_edge

    @property
    def num_relation(self):
        """Number of relations."""
        return self.graph.num_relation

@R.register("datasets.CoDExSmall")
class CoDExSmall(CoDEx):
    """
    #node: 2034
    #edge: 36543
    #relation: 42
    """
    url = "https://zenodo.org/record/4281094/files/codex-s.tar.gz"
    md5 = "63cd8186fc2aeddc154e20cf4a10087e"
    name = "CoDExSmall"

    def __init__(self, **kwargs):
        super(CoDExSmall, self).__init__(size='s', **kwargs)

@R.register("datasets.CoDExMedium")
class CoDExMedium(CoDEx):
    """
    #node: 17050
    #edge: 206205
    #relation: 51
    """
    url = "https://zenodo.org/record/4281094/files/codex-m.tar.gz"
    md5 = "43e561cfdca1c6ad9cc2f5b1ca4add76"
    name = "CoDExMedium"
    def __init__(self, **kwargs):
        super(CoDExMedium, self).__init__(size='m', **kwargs)

@R.register("datasets.CoDExLarge")
class CoDExLarge(CoDEx):
    """
    #node: 77951
    #edge: 612437
    #relation: 69
    """
    url = "https://zenodo.org/record/4281094/files/codex-l.tar.gz"
    md5 = "9a10f4458c4bd2b16ef9b92b677e0d71"
    name = "CoDExLarge"
    def __init__(self, **kwargs):
        super(CoDExLarge, self).__init__(size='l', **kwargs)

class ILPC2022Inductive(torch_data.Dataset, core.Configurable):

    url = ""
    md5 = ""
    name = ""

    def __init__(self, path, size="small", verbose=1, **kwargs):
        super(ILPC2022Inductive, self).__init__()

        path = os.path.expanduser(path)
        path = os.path.join(path, f"ilpc-{size}")
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)

        print("dataset name:", self.name)

        train_file = utils.extract(zip_file, f"pykeen-ilpc2022-c5ea003/data/{size}/train.txt")
        inference_file = utils.extract(zip_file, f"pykeen-ilpc2022-c5ea003/data/{size}/inference.txt")
        inference_valid_file = utils.extract(zip_file, f"pykeen-ilpc2022-c5ea003/data/{size}/inference_validation.txt")
        inference_test_file = utils.extract(zip_file, f"pykeen-ilpc2022-c5ea003/data/{size}/inference_test.txt")

        train_results = self.load_node(train_file)
        inference_results = self.load_node(inference_file)
        valid_results = self.load_node(inference_valid_file, 
                        inference_results["inv_entity_vocab"], inference_results["inv_rel_vocab"])
        test_results = self.load_node(inference_test_file,
                        inference_results["inv_entity_vocab"], inference_results["inv_rel_vocab"])

        num_train_node, num_inf_node = train_results["num_node"], inference_results["num_node"]
        num_rel = train_results["num_relation"]
        assert train_results["num_relation"] == inference_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]
        inference_graph = inference_results["triplets"]

        self.train_graph = data.Graph(train_triplets, num_node=num_train_node, num_relation=num_rel)
        self.valid_graph = data.Graph(inference_graph, num_node=num_inf_node, num_relation=num_rel)
        self.test_graph = data.Graph(inference_graph, num_node=num_inf_node, num_relation=num_rel)
        self.graph = self.train_graph
        self.inductive_graph = data.Graph(inference_graph + valid_triplets + test_triplets, num_node=num_inf_node, num_relation=num_rel)
        
        self.triplets = torch.tensor(train_triplets + valid_triplets + test_triplets)

        self.num_samples = [len(train_triplets),
                                len(valid_triplets), len(test_triplets)]
        self.transform = None

    def load_node(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split()
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }

    def __getitem__(self, index):
        return self.triplets[index]
        
    def __len__(self):
        return self.graph.num_edge

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, list(range(offset, offset + num_sample)))
            splits.append(split)
            offset += num_sample
        return splits

    @property
    def num_entity(self):
        """Number of entities."""
        return self.graph.num_node

    @property
    def num_triplet(self):
        """Number of triplets."""
        return self.graph.num_edge

    @property
    def num_relation(self):
        """Number of relations."""
        return self.graph.num_relation


@R.register("datasets.ILPC2022LargeInductive")
class ILPC2022LargeInductive(ILPC2022Inductive):

    #url = "https://storage.cloud.google.com/public_dataset_xinyu/ilpc2022large.zip?_ga=2.18504958.-634376353.1663700246"
    #md5 = "82e3b3bb69a7dfa5cc9fe61595b8cd85"
    url = "https://zenodo.org/record/6321299/files/pykeen/ilpc2022-v1.0.zip"
    md5 = "5f3da4b0812c809a976e40b7bae72ed1"
    name = "ILPC2022LargeInductive"

    def __init__(self, **kwargs):
        super(ILPC2022LargeInductive, self).__init__(size="large", **kwargs)

@R.register("datasets.ILPC2022SmallInductive")
class ILPC2022SmallInductive(ILPC2022Inductive):

    # url = "https://storage.cloud.google.com/public_dataset_xinyu/ilpc2022small.zip?_ga=2.18504958.-634376353.1663700246"
    # md5 = "246aeead0240ed5a133585902830f27a"
    url = "https://zenodo.org/record/6321299/files/pykeen/ilpc2022-v1.0.zip"
    md5 = "5f3da4b0812c809a976e40b7bae72ed1"
    name = "ILPC2022SmallInductive"

    def __init__(self, **kwargs):
        super(ILPC2022SmallInductive, self).__init__(size="small", **kwargs)


class InductiveKnowledgeGraphDataset(data.KnowledgeGraphDataset):

    def load_inductive_tsvs(self, transductive_files, inductive_files, merge_valid_test=False, use_inductive_valid=False, verbose=0):
        assert len(transductive_files) == len(inductive_files) == 3
        inv_transductive_vocab = {}
        inv_inductive_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in transductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_transductive_vocab:
                        inv_transductive_vocab[h_token] = len(inv_transductive_vocab)
                    h = inv_transductive_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_transductive_vocab:
                        inv_transductive_vocab[t_token] = len(inv_transductive_vocab)
                    t = inv_transductive_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in inductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_inductive_vocab:
                        inv_inductive_vocab[h_token] = len(inv_inductive_vocab)
                    h = inv_inductive_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_inductive_vocab:
                        inv_inductive_vocab[t_token] = len(inv_inductive_vocab)
                    t = inv_inductive_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        transductive_vocab, inv_transductive_vocab = self._standarize_vocab(None, inv_transductive_vocab)
        inductive_vocab, inv_inductive_vocab = self._standarize_vocab(None, inv_inductive_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)

        #ipdb.set_trace()

        self.train_graph = data.Graph(triplets[:num_samples[0]],
                                      num_node=len(transductive_vocab), num_relation=len(relation_vocab))
        self.test_graph = data.Graph(triplets[sum(num_samples[:3]): sum(num_samples[:4])],
                                        num_node=len(inductive_vocab), num_relation=len(relation_vocab))
        if not use_inductive_valid:
            self.valid_graph = copy.copy(self.train_graph)
        else:
            self.valid_graph = copy.copy(self.test_graph)
        self.graph = data.Graph(triplets[:sum(num_samples[:3])],
                                num_node=len(transductive_vocab), num_relation=len(relation_vocab))
        self.inductive_graph = data.Graph(triplets[sum(num_samples[:3]):],
                                          num_node=len(inductive_vocab), num_relation=len(relation_vocab))
        if merge_valid_test:
            if use_inductive_valid: # inductive.train + inductive.(valid&test)
                self.triplets = torch.tensor(triplets[:sum(num_samples[:1])] 
                                             + triplets[sum(num_samples[:3]):sum(num_samples[:4])] 
                                             + triplets[sum(num_samples[:4]):])
                self.num_samples = num_samples[:1] + num_samples[3:4] + [sum(num_samples[4:])]
            else: # transductive.valid + inductive.(valid&test)
                self.triplets = torch.tensor(triplets[:sum(num_samples[:2])] 
                                             + triplets[sum(num_samples[:4]):])
                self.num_samples = num_samples[:2] + [sum(num_samples[4:])]
        else:
            if use_inductive_valid: # inductive.valid + inductive.test
                self.triplets = torch.tensor(triplets[:sum(num_samples[:1])] 
                                             + triplets[sum(num_samples[:4]):sum(num_samples[:5])] 
                                             + triplets[sum(num_samples[:5]):])
                self.num_samples = num_samples[:1] + num_samples[4:5] + [sum(num_samples[5:])]
            else: # transductive.valid + inductive.test
                self.triplets = torch.tensor(triplets[:sum(num_samples[:2])] 
                                             + triplets[sum(num_samples[:5]):])
                self.num_samples = num_samples[:2] + [sum(num_samples[5:])]
        
        self.transductive_vocab = transductive_vocab
        self.inductive_vocab = inductive_vocab
        self.relation_vocab = relation_vocab
        self.inv_transductive_vocab = inv_transductive_vocab
        self.inv_inductive_vocab = inv_inductive_vocab
        self.inv_relation_vocab = inv_relation_vocab

    def __getitem__(self, index):
        return self.triplets[index]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.FB15k237Inductive")
class FB15k237Inductive(InductiveKnowledgeGraphDataset):

    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/test.txt",
    ]

    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", merge_valid_test=True, use_inductive_valid=False, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        transductive_files = []
        for url in self.transductive_urls:
            url = url % version
            save_file = "fb15k237_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            transductive_files.append(txt_file)
        inductive_files = []
        for url in self.inductive_urls:
            url = url % version
            save_file = "fb15k237_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            inductive_files.append(txt_file)

        self.load_inductive_tsvs(transductive_files, inductive_files,
                                 merge_valid_test=merge_valid_test, 
                                 use_inductive_valid=use_inductive_valid, verbose=verbose)


@R.register("datasets.WN18RRInductive")
class WN18RRInductive(InductiveKnowledgeGraphDataset):

    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/test.txt",
    ]

    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", merge_valid_test=True, use_inductive_valid=False, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        transductive_files = []
        for url in self.transductive_urls:
            url = url % version
            save_file = "wn18rr_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            transductive_files.append(txt_file)
        inductive_files = []
        for url in self.inductive_urls:
            url = url % version
            save_file = "wn18rr_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            inductive_files.append(txt_file)

        self.load_inductive_tsvs(transductive_files, inductive_files,
                                 merge_valid_test=merge_valid_test, 
                                 use_inductive_valid=use_inductive_valid, verbose=verbose)
        
@R.register("datasets.NELLInductive")
class NELLInductive(InductiveKnowledgeGraphDataset):

    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/test.txt",
    ]

    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", merge_valid_test=True, use_inductive_valid=False, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        transductive_files = []
        for url in self.transductive_urls:
            url = url % version
            save_file = "nell_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            transductive_files.append(txt_file)
        inductive_files = []
        for url in self.inductive_urls:
            url = url % version
            save_file = "nell_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            inductive_files.append(txt_file)

        self.load_inductive_tsvs(transductive_files, inductive_files,
                                 merge_valid_test=merge_valid_test, 
                                 use_inductive_valid=use_inductive_valid, verbose=verbose)


@R.register("datasets.ConceptNet100k")
class ConceptNet100k(data.KnowledgeGraphDataset):

    urls = [
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/train",
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/valid",
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/test",
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url in self.urls:
            save_file = "cn100k_%s" % os.path.basename(url)
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            txt_files.append(txt_file)

        self.load_tsvs(txt_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.ATOMIC")
class ATOMIC(data.KnowledgeGraphDataset):

    folder = 'Atomic'
    # the zip has to be downloaded from https://drive.google.com/file/d/1X7uxP95GyRt42z2xP0I_tPkE22XAQ-_J/view
    # and unzipped in the kg-datasets
    # didn't include downloading because of bringing the gdown dependency
    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = os.path.join(path, self.folder)

        dump_files = []
        dump_files.append(os.path.join(self.path, f"train"))
        dump_files.append(os.path.join(self.path, f"valid"))
        dump_files.append(os.path.join(self.path, f"test"))

        self.load_tsvs(dump_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits
    

@R.register("datasets.AristoV4")
class AristoV4(data.KnowledgeGraphDataset):

    url = "https://zenodo.org/record/5942560/files/aristo-v4.zip"
    md5 = "4e0e9be2808b9b43194494ba61f8f97b"

    # TODO clean the valid/test sets from triples having unseen (from train) entities
    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        zip_file = os.path.join(path, os.path.basename(self.url))
        if not os.path.exists(zip_file):
            zip_file = utils.download(self.url, path, md5=self.md5)
        txt_files = [utils.extract(zip_file, split) for split in ['train', 'valid', 'test']]
        self.load_tsvs(txt_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.JointDataset")
class JointDataset(data.KnowledgeGraphDataset):

    datasets_map = {
        'FB15k237': datasets.FB15k237,
        'WN18RR': datasets.WN18RR,
        'YAGO310': datasets.YAGO310,
        'CoDExSmall': CoDExSmall,
        'CoDExMedium': CoDExMedium,
        'CoDExLarge': CoDExLarge,
        'ConceptNet100k': ConceptNet100k,
        'AristoV4': AristoV4,
        'ATOMIC': ATOMIC,
        # TODO: doesn't really work with inductive datasets for now (need more wrangling with train/inference graphs)
        'ILPCSmall': ILPC2022SmallInductive,
        'ILPCLarge': ILPC2022LargeInductive,
        'FB15k237Inductive': partial(FB15k237Inductive, merge_valid_test=False, use_inductive_valid=True),
        'WN18RRInductive': partial(WN18RRInductive, merge_valid_test=False, use_inductive_valid=True)
    }

    def __init__(self, path, graphs, verbose=1, *args, **kwargs):
        super(JointDataset, self).__init__(*args, **kwargs)

        # Initialize all specified KGs
        self.graphs = [
            self.datasets_map[dataset](path=path, verbose=verbose) for dataset in graphs
        ]
        self.graph_names = graphs
        # Total number of samples obtained from iterating over all graphs
        self.num_samples = [sum(k) for k in zip(*[graph.num_samples for graph in self.graphs])]
        self.valid_samples = [torch.cumsum(torch.tensor(k).flatten(), dim=0) for k in zip([graph.num_samples for graph in self.graphs])]
    
    def __getitem__(self, index):
        # TODO find where to specify the graph id
        # graph_id = torch.randint(0, len(self.graphs))
        # graph = self.graphs[graph_id]
        # return graph.edge_list[index]
        # send a dummy entry, we'll be sampling edges in the collator function
        return torch.zeros(1,1)

    def __len__(self):
        return sum([graph.num_edge for graph in self.graphs])

    def split(self):
        offset = 0
        splits = []
        # train - essentially, one list of IDs
        splits.append(torch_data.Subset(self, range(offset, offset + self.num_samples[0])))
        # for validation and test, keep slices for each graph, one list of edge IDs per graph
        splits.append({self.graph_names[i]: (i, l[0:2]) for i,l in enumerate(self.valid_samples)})
        splits.append({self.graph_names[i]: (i, l[1:]) for i,l in enumerate(self.valid_samples)})
        # for num_sample in self.num_samples:
        #     split = torch_data.Subset(self, range(offset, offset + num_sample))
        #     splits.append(split)
        #     offset += num_sample
        return splits
    
    @property
    def num_entity(self):
        """Number of entities in the joint graph"""
        return sum(graph.num_entity for graph in self.graphs)

    @property
    def num_triplet(self):
        """Number of triplets in the joint graph"""
        return sum(graph.num_triplet for graph in self.graphs)

    @property
    def num_relation(self):
        """Number of relations in the joint graph"""
        return sum(graph.num_relation for graph in self.graphs)
    
class IngramInductive(torch_data.Dataset, core.Configurable):

    def __init__(self, path, version, **kwargs):
        path = os.path.expanduser(path)
        path = os.path.join(path, f"{self.prefix}-{version}")
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        dump_files = []
        for url in self.urls:
            url = url % version
            save_file = "%s-%s_%s" % (self.prefix, version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            dump_files.append(txt_file)

        self.load_files(dump_files)

    def load_files(self, dump_files):

        train_results = self.load_file(dump_files[0], {}, {})
        inference_results = self.load_file(dump_files[1], {}, {})
        valid_results = self.load_file(dump_files[2], 
                                       inference_results["inv_entity_vocab"],
                                       inference_results["inv_rel_vocab"])
        test_results = self.load_file(dump_files[3], 
                                      inference_results["inv_entity_vocab"],
                                      inference_results["inv_rel_vocab"])

        num_train_node, num_inf_node = train_results["num_node"], inference_results["num_node"]
        num_train_rel = train_results["num_relation"]
        num_inference_rel = inference_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]
        inference_graph = inference_results["triplets"]

        self.train_graph = data.Graph(train_triplets, num_node=num_train_node, num_relation=num_train_rel)
        self.valid_graph = data.Graph(inference_graph, num_node=num_inf_node, num_relation=num_inference_rel)
        self.test_graph = data.Graph(inference_graph, num_node=num_inf_node, num_relation=num_inference_rel)
        self.graph = self.train_graph
        self.inductive_graph = data.Graph(inference_graph + valid_triplets + test_triplets, num_node=num_inf_node, num_relation=num_inference_rel)
        
        self.triplets = torch.tensor(train_triplets + valid_triplets + test_triplets)

        self.num_samples = [len(train_triplets),
                                len(valid_triplets), len(test_triplets)]
        self.transform = None
    
    def load_file(self, fname, inv_entity_vocab={}, inv_rel_vocab={}):
        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(fname, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split()
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }

    def __getitem__(self, index):
        return self.triplets[index]
        
    def __len__(self):
        return self.graph.num_edge

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, list(range(offset, offset + num_sample)))
            splits.append(split)
            offset += num_sample
        return splits

    @property
    def num_entity(self):
        """Number of entities."""
        return self.graph.num_node

    @property
    def num_triplet(self):
        """Number of triplets."""
        return self.graph.num_edge

    @property
    def num_relation(self):
        """Number of relations."""
        return self.graph.num_relation

@R.register("datasets.FBIngram")
class FBIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/test.txt",
    ]
    prefix = "fb"

    def __init__(self, **kwargs):
        super(FBIngram, self).__init__(**kwargs)


@R.register("datasets.WKIngram")
class WKIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/test.txt",
    ]
    prefix = "wk"

    def __init__(self, **kwargs):
        super(WKIngram, self).__init__(**kwargs)

@R.register("datasets.NLIngram")
class NLIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/test.txt",
    ]
    prefix = "nl"

    def __init__(self, **kwargs):
        super(NLIngram, self).__init__(**kwargs)


class MTDEAInductive(IngramInductive):

    # the datasets aren't available online, so should be put physically in <path>
    def __init__(self, path, version, **kwargs):
        # path = os.path.expanduser(path)
        # path = os.path.join(path, f"{self.prefix}-{version}")
        # if not os.path.exists(path):
        #     os.makedirs(path)
        if version is not None:
            assert version in self.versions, f"unknown version {version} for {self.folder}, available: {self.versions}"
        path = os.path.expanduser(path)
        self.path = os.path.join(path, self.folder)

        dump_files = []
        prefix = self.prefix % version if version is not None else self.prefix
        dump_files.append(os.path.join(self.path, f"{prefix}-trans/train.txt"))
        dump_files.append(os.path.join(self.path, f"{prefix}-trans/valid.txt"))
        dump_files.append(os.path.join(self.path, f"{prefix}-ind/observe.txt"))
        dump_files.append(os.path.join(self.path, f"{prefix}-ind/test.txt"))

        self.load_files(dump_files)

    def load_files(self, dump_files):

        train_results = self.load_file(dump_files[0], {}, {})
        inference_results = self.load_file(dump_files[2], {}, {})
        valid_results = self.load_file(dump_files[1], 
                                       train_results["inv_entity_vocab"],
                                       train_results["inv_rel_vocab"],
                                       limit_vocab=True)
        test_results = self.load_file(dump_files[3], 
                                      inference_results["inv_entity_vocab"],
                                      inference_results["inv_rel_vocab"])

        num_train_node, num_inf_node = train_results["num_node"], inference_results["num_node"]
        num_train_rel = train_results["num_relation"]
        num_inference_rel = inference_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]
        inference_graph = inference_results["triplets"]

        # In all MTDEA datasets validation triples are the part of the training graph, test triples - of the inference graph
        self.train_graph = data.Graph(train_triplets, num_node=num_train_node, num_relation=num_train_rel)
        self.valid_graph = data.Graph(train_triplets, num_node=valid_results["num_node"], num_relation=num_train_rel)
        self.test_graph = data.Graph(inference_graph, num_node=test_results["num_node"], num_relation=num_inference_rel)
        self.graph = data.Graph(train_triplets + valid_triplets, num_node=valid_results["num_node"], num_relation=num_train_rel)
        self.inductive_graph = data.Graph(inference_graph + test_triplets, num_node=test_results["num_node"], num_relation=num_inference_rel)
        
        self.triplets = torch.tensor(train_triplets + valid_triplets + test_triplets)

        self.num_samples = [len(train_triplets),
                                len(valid_triplets), len(test_triplets)]
        self.transform = None
    
    def load_file(self, fname, inv_entity_vocab={}, inv_rel_vocab={}, limit_vocab=False):
        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)
        
        # limit_vocab is for dropping triples with unseen head/tail not seen in the main entity_vocab
        # can be used for FBNELL, other datasets seem to be ok and share num_nodes in the inference graph and test triples  
        with open(fname, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split()
                if u not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    if limit_vocab:
                        continue
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    

@R.register("datasets.FBNELL")
class FBNELL(MTDEAInductive):

    folder = "FBNELL"
    prefix = "FBNELL_v1"

    def __init__(self, **kwargs):
        kwargs.pop("version")
        super(FBNELL, self).__init__(version=None, **kwargs)

@R.register("datasets.Metafam")
class Metafam(MTDEAInductive):

    folder = "Metafam"
    prefix = "Metafam"

    def __init__(self, **kwargs):
        kwargs.pop("version")
        super(Metafam, self).__init__(version=None, **kwargs)

@R.register("datasets.WikiTopicsMT1")
class WikiTopicsMT1(MTDEAInductive):

    folder = "WikiTopics-MT1"
    prefix = "wikidata_%sv1"
    versions = ['mt', 'health', 'tax']

    def __init__(self, **kwargs):
        assert kwargs['version'] in self.versions, f"unknown version {kwargs['version']}, available: {self.versions}"
        super(WikiTopicsMT1, self).__init__(**kwargs)

@R.register("datasets.WikiTopicsMT2")
class WikiTopicsMT2(MTDEAInductive):

    folder = "WikiTopics-MT2"
    prefix = "wikidata_%sv1"
    versions = ['mt2', 'org', 'sci']

    def __init__(self, **kwargs):
        super(WikiTopicsMT2, self).__init__(**kwargs)

@R.register("datasets.WikiTopicsMT3")
class WikiTopicsMT3(MTDEAInductive):

    folder = "WikiTopics-MT3"
    prefix = "wikidata_%sv2"
    versions = ['mt3', 'art', 'infra']

    def __init__(self, **kwargs):
        super(WikiTopicsMT3, self).__init__(**kwargs)

@R.register("datasets.WikiTopicsMT4")
class WikiTopicsMT4(MTDEAInductive):

    folder = "WikiTopics-MT4"
    prefix = "wikidata_%sv2"
    versions = ['mt4', 'sci', 'health']

    def __init__(self, **kwargs):
        super(WikiTopicsMT4, self).__init__(**kwargs)

class BMInductive(IngramInductive):

    def __init__(self, path, version, **kwargs):
        assert version in list(self.versions.keys()), f"unknown version {version}, available: {list(self.versions.keys())}"
        path = os.path.expanduser(path)
        path = os.path.join(path, f"{self.prefix}-{version}")
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        dump_files = []
        for url in self.urls:
            url = url % self.versions[version]
            save_file = "%s-%s_%s" % (self.prefix, version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            dump_files.append(txt_file)

        self.load_files(dump_files)

    def load_files(self, dump_files):

        train_results = self.load_file(dump_files[0], {}, {})
        inference_results = self.load_file(dump_files[1], {}, {})
        valid_results = self.load_file(dump_files[2], 
                                       train_results["inv_entity_vocab"],
                                       train_results["inv_rel_vocab"])
        test_results = self.load_file(dump_files[3], 
                                      inference_results["inv_entity_vocab"],
                                      inference_results["inv_rel_vocab"])

        num_train_node, num_inf_node = train_results["num_node"], inference_results["num_node"]
        num_train_rel = train_results["num_relation"]
        num_inference_rel = inference_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]
        inference_graph = inference_results["triplets"]

        self.train_graph = data.Graph(train_triplets, num_node=num_train_node, num_relation=num_train_rel)
        self.valid_graph = data.Graph(train_triplets, num_node=valid_results['num_node'], num_relation=num_train_rel)
        self.test_graph = data.Graph(inference_graph, num_node=test_results['num_node'], num_relation=num_inference_rel)
        self.graph = data.Graph(train_triplets + valid_triplets, num_node=valid_results['num_node'], num_relation=num_train_rel)
        self.inductive_graph = data.Graph(inference_graph + test_triplets, num_node=test_results['num_node'], num_relation=num_inference_rel)
        
        self.triplets = torch.tensor(train_triplets + valid_triplets + test_triplets)

        self.num_samples = [len(train_triplets),
                                len(valid_triplets), len(test_triplets)]
        self.transform = None
    
    
@R.register("datasets.HamaguchiBM")
class HamaguchiBM(BMInductive):

    urls = [
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/train/train.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/test/test-graph.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/train/valid.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/test/test-fact.txt",
    ]
    prefix = "bm"
    versions = {
        '1k': "Hamaguchi-BM_both-1000",
        '3k': "Hamaguchi-BM_both-3000",
        '5k': "Hamaguchi-BM_both-5000",
        'indigo': "INDIGO-BM" 
    }

    def __init__(self, **kwargs):
        super(HamaguchiBM, self).__init__(**kwargs)

    
@R.register("datasets.DBpedia50k")
class DBpedia50k(data.KnowledgeGraphDataset):

    urls = [
        "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50/train.txt",
        "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50/valid.txt",
        "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50/test.txt",
    ]

    # Weird dataset, valid/test set is 400/10000 triples, test triples have unseen entities
    # eval should be head or tail
    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url in self.urls:
            save_file = "dbp50k_%s" % os.path.basename(url)
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            txt_files.append(txt_file)

        self.load_tsvs(txt_files, verbose=verbose)
    
    def load_tsvs(self, tsv_files, verbose=0):
        """
        Load the dataset from multiple tsv files.

        Parameters:
            tsv_files (list of str): list of file names
            verbose (int, optional): output verbose level
        """
        inv_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for tsv_file in tsv_files:
            with open(tsv_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % tsv_file, utils.get_line_count(tsv_file))

                num_sample = 0
                for tokens in reader:
                    h_token, t_token, r_token = tokens
                    if h_token not in inv_entity_vocab:
                        inv_entity_vocab[h_token] = len(inv_entity_vocab)
                    h = inv_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_entity_vocab:
                        inv_entity_vocab[t_token] = len(inv_entity_vocab)
                    t = inv_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        self.load_triplet(triplets, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab)
        self.num_samples = num_samples

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits
    
@R.register("datasets.DBpedia100k")
class DBpedia100k(data.KnowledgeGraphDataset):

    urls = [
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_train.txt",
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_valid.txt",
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_test.txt",
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url in self.urls:
            save_file = "dbp100k%s" % os.path.basename(url)
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            txt_files.append(txt_file)

        self.load_tsvs(txt_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits
    
class SparserKG(DBpedia50k):

    # 5 datasets based on FB/NELL/WD, introduced in https://github.com/THU-KEG/DacKGR
    # inheriting from DBpedia50k because dumps are in the format (h, t, r)

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = os.path.join(path, self.folder)

        dump_files = []
        dump_files.append(os.path.join(self.path, f"train.triples"))
        dump_files.append(os.path.join(self.path, f"dev.triples"))
        dump_files.append(os.path.join(self.path, f"test.triples"))

        self.load_tsvs(dump_files, verbose=verbose)
    

@R.register("datasets.WDsinger")
class WDsinger(SparserKG):   

    folder = "WD-singer"

@R.register("datasets.NELL23k")
class NELL23k(SparserKG):   

    folder = "NELL23K"

@R.register("datasets.FB15k237_10")
class FB15k237_10(SparserKG):   

    folder = "FB15K-237-10"

@R.register("datasets.FB15k237_20")
class FB15k237_20(SparserKG):   

    folder = "FB15K-237-20"

@R.register("datasets.FB15k237_50")
class FB15k237_50(SparserKG):   

    folder = "FB15K-237-50"

@R.register("datasets.NELL995")
class NELL995(data.KnowledgeGraphDataset):

    # from the RED-GNN paper https://github.com/LARS-research/RED-GNN/tree/main/transductive/data/nell
    # the OG dumps were found to have test set leakages
    # training set is made out of facts+train files, so we sum up their num_samples to build one chunk

    urls = [
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/facts.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/train.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/valid.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/test.txt",
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url in self.urls:
            save_file = "nell995_%s" % os.path.basename(url)
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            txt_files.append(txt_file)

        self.load_tsvs(txt_files, verbose=verbose)
        self.num_samples = [self.num_samples[0]+self.num_samples[1], self.num_samples[2], self.num_samples[3]]


    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits
    

class UrbanKG(data.KnowledgeGraphDataset):
       
    folder = 'UrbanKG_%s'

    # Download raw dumps from https://drive.google.com/drive/folders/1LvR2ZUnx6R6C-CxzmBDbQoNjDSQfE5YV
    # place and unzip manually, autodownloading as a later TODO

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = os.path.join(path, self.folder % self.suffix)

        dump_files = []
        dump_files.append(os.path.join(self.path, f"train_{self.suffix}.txt"))
        dump_files.append(os.path.join(self.path, f"valid_{self.suffix}.txt"))
        dump_files.append(os.path.join(self.path, f"test_{self.suffix}.txt"))

        self.load_tsvs(dump_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits
    

@R.register("datasets.UUKG_NYC")
class UUKG_NYC(UrbanKG):

    suffix = "NYC"

@R.register("datasets.UUKG_CHI")
class UUKG_CHI(UrbanKG):

    suffix = "CHI"