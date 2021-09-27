import argparse
import json
import os
import pickle
from copy import deepcopy
import sys
import timeit
import numpy as np
import time
from fairseq.utils import import_user_module
from fairseq.tokenizer import tokenize_line
from tokenizer import * 
from dptree.tree_process import *
from binarization import *
from tokenizer import * 
from data.dptree_index_dataset import *
from fairseq import tasks,options
from fairseq.data import dictionary
from collections import Counter
from nstack_tokenizer import NstackTreeTokenizer
 
import sys
from asdl.hypothesis import *
from asdl.lang.py3.py3_transition_system import python_ast_to_asdl_ast, asdl_ast_to_python_ast, Python3TransitionSystem
from asdl.transition_system import *
from components.action_info import get_action_infos
from components.dataset_new import Example
from components.vocab import Vocab, VocabEntry 
from datasets.conala.evaluator import ConalaEvaluator
from datasets.conala.util import *


#imports 

from copy import deepcopy
import traceback



class CusCoreNLPParser(CoreNLPParser):
    def api_call(self, data, properties=None, timeout=18000000, lang=None):
        if properties is None:
            properties = {'parse.binaryTrees': "true"}
        return super().api_call(data, properties, timeout)
    @classmethod
    def build_parser(cls, port=9001):
        port = str(port)
        return cls(url=f'http://localhost:{port}')

core_parser = CusCoreNLPParser.build_parser(9000)

SPECIAL_CHAR = {'&apos;': "'", '&apos;s': "'s", '&quot;': '"', '&#91;': '[',
                '&#93;': "]", '&apos;@@': "'@@", '&apos;t': "'t",
                '&amp;': "&", '&apos;ll': "'ll", '&apos;ve': "'ve",
                '&apos;m': "'m", '&apos;re': "'re", '&apos;d': "'d",
                '&#124;': "|", '&gt;': ">", '&lt;': "<"}
SPECIAL_CHAR_MBACK = {v: k for k, v in SPECIAL_CHAR.items()}
SPECIAL_CHAR_MBACK['-LSB-'] = '&#91;'
SPECIAL_CHAR_MBACK['-RSB-'] = '&#93;'
SPECIAL_CHAR_MBACK['-LRB-'] = "("
SPECIAL_CHAR_MBACK['-RRB-'] = ")"
SPECIAL_CHAR_MBACK["''"] = "&quot;"

def replace_special_character(string):
    new_string = deepcopy(string)
    # new_string = new_string.replace(u"）", ")").replace(u"（", "(")
    new_string = new_string.replace(")", u"）").replace("(", u"（")

    list_string = new_string.split(" ")
    new_list = deepcopy(list_string)
    for i in range(len(list_string)):
        for k, v in SPECIAL_CHAR.items():
            if k in list_string[i]:
                new_list[i] = list_string[i].replace(k, v)
    return " ".join(new_list)

def remap_chars(tree):
    for i in range(len(tree.leaves())):
        if tree.leaves()[i] in SPECIAL_CHAR_MBACK:
            tree[tree.leaf_treeposition(i)] = SPECIAL_CHAR_MBACK[tree.leaves()[i]]


def parse_string(parser, bpe_string, unify_tree=False):
    word_string_nobpe = bpe_string.replace("@@ ", "")
    word_string = replace_special_character(word_string_nobpe)
    try:
        tree_strings = list(parser.parse_text(word_string))
    
    except Exception as e:
        try:
            print(f'Try bpe version')
            print("the original string is ", bpe_string)
            tree_strings = list(parser.parse_text(bpe_string))
        except Exception as ee:
            print('we get here ')
            print(f'Failed.')
            print(f'[Ori]: {bpe_string}')
            print(f'[Proc]: {word_string}')
            traceback.print_stack()
            raise ee
    
    token_set = set()
    parse_strings = []
    befores = []
    afters = []
    
    for tree_s in tree_strings:
        before = deepcopy(tree_s)
        remap_chars(tree_s)
        after = deepcopy(tree_s)
        parse_string = ' '.join(str(tree_s).split())
        token_set = token_set.union(set(after.leaves()))
        parse_strings.append(parse_string)
        befores.append(before)
        afters.append(after)
    # print(parse_strings)
    return parse_strings, [befores, afters], token_set



def preprocess_conala_dataset(train_file,test_file, grammar_file, src_freq=3, code_freq=3,
                              mined_data_file=None, api_data_file=None,
                              vocab_size=5000, num_mined=0, out_dir='data/conala',args=None):
    np.random.seed(1234)

    asdl_text = open(grammar_file).read()
    grammar = ASDLGrammar.from_text(asdl_text)
    transition_system = Python3TransitionSystem(grammar)

    print('process gold training data...')
    #this is basically where the magic should happen
    train_examples = preprocess_dataset(train_file, name='train', transition_system=transition_system,grammar=grammar,args=args)
    
    full_train_examples = train_examples
    np.random.shuffle(train_examples)
    dev_examples = train_examples[:200]
    train_examples = train_examples[200:]

    mined_examples = []
    api_examples = []
    if mined_data_file and num_mined > 0:
        print("use mined data: ", num_mined)
        print("from file: ", mined_data_file)
        mined_examples = preprocess_dataset(mined_data_file, name='mined', transition_system=transition_system,
                                            firstk=num_mined,grammar=grammar)
        pickle.dump(mined_examples, open(os.path.join(out_dir, 'mined_{}.bin'.format(num_mined)), 'wb'))

    if api_data_file:
        print("use api docs from file: ", api_data_file)
        name = os.path.splitext(os.path.basename(api_data_file))[0]
        api_examples = preprocess_dataset(api_data_file, name='api', transition_system=transition_system,grammar=grammar)
        pickle.dump(api_examples, open(os.path.join(out_dir, name + '.bin'), 'wb'))

    if mined_examples and api_examples:
        pickle.dump(mined_examples + api_examples, open(os.path.join(out_dir, 'pre_{}_{}.bin'.format(num_mined, name)), 'wb'))

    # combine to make vocab
    train_examples += mined_examples
    train_examples += api_examples
    # print (f'{len(pretrain_examples)} pretraining instances', file=sys.stderr)
    print(f'{len(train_examples)} training instances', file=sys.stderr)
    print(f'{len(dev_examples)} dev instances', file=sys.stderr)

    print('process testing data...')
    test_examples = preprocess_dataset(test_file, name='test', transition_system=transition_system,grammar=grammar,args=args)
    print(f'{len(test_examples)} testing instances', file=sys.stderr)

    #we will have a new src vocab
    #TODO: how does the new vocab incorporate pos tags.
    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_examples], size=vocab_size,
                                       freq_cutoff=src_freq)

    primitive_tokens = [list(map(lambda a: a.action.token,
                        filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions)))\
                    for e in train_examples]
    print('len of primitive, ',len(primitive_tokens),len(train_examples))

    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=vocab_size, freq_cutoff=code_freq)
    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=primitive_vocab)
    print('generated vocabulary %s' % repr(vocab))

    action_lens = [len(e.tgt_actions) for e in train_examples]

    pickle.dump(train_examples, open(os.path.join(out_dir, 'train.bin'), 'wb'))
    pickle.dump(dev_examples, open(os.path.join(out_dir, 'dev.bin'), 'wb'))
    pickle.dump(test_examples, open(os.path.join(out_dir, 'test.bin'), 'wb'))


    print('Max action len: %d' % max(action_lens), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_lens), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_lens))), file=sys.stderr)
    
    
    vocab_name = 'vocab.bin'
    pickle.dump(vocab, open(os.path.join(out_dir, vocab_name), 'wb'))
    


def read(file):
    datapath=json.load(open(file))
    dataset=[]
    for i,data in enumerate(datapath):
        try:
            tree=ast.parse(read_file_to_string(datapath[i]['filename']))
            dataset.append(tree)
        except:
            pass

def read_file_to_string(filename):
    f = open(filename, 'rt')
    s = f.read()
    f.close()
    return s


def preprocess_dataset(file_path, transition_system, name='train', firstk=None,grammar=None,args=None):
    try:
        dataset = json.load(open(file_path))
    except:
        dataset = [json.loads(jline) for jline in open(file_path).readlines()]
    if firstk:
        dataset = dataset[:firstk]
    examples = []
    f = open(file_path + '.debug', 'w')
    skipped_list = []
    for i, example_json in enumerate(dataset):
       
        example_dict = preprocess_example(example_json,args=args)
        try:
            python_ast = ast.parse(example_dict['canonical_snippet'])
        except:
            print (file_path)
            continue
        text=example_dict['intent_tokens']
        canonical_code = astor.to_source(python_ast).strip()
        tgt_ast = python_ast_to_asdl_ast(python_ast, transition_system.grammar)
        
        actions,input_actions,tgt_actions = transition_system.get_actions(asdl_ast=tgt_ast,text=text,grammar=grammar)


        tgt_action_infos = get_action_infos(example_dict['intent_tokens'], tgt_actions,pretrain=False)
        # prinnue
        example = Example(tgt_actions=tgt_action_infos,
                          input_actions=input_actions,
                          idx=f'{i}-{example_json["question_id"]}',
                          src_sent=example_dict['intent_tokens'],
                          leaves_nodes=example_dict['leaves_nodes'],
                          tgt_code=canonical_code,
                          tgt_ast=tgt_ast,
                          meta=dict(example_dict=example_json,
                                    slot_map=example_dict['slot_map']))

        examples.append(example)

        # log!
        f.write(f'Example: {example.idx}\n')
        if 'rewritten_intent' in example.meta['example_dict']:
            f.write(f"Original Utterance: {example.meta['example_dict']['rewritten_intent']}\n")
        else:
            f.write(f"Original Utterance: {example.meta['example_dict']['intent']}\n")
        f.write(f"Original Snippet: {example.meta['example_dict']['snippet']}\n")
        f.write(f"\n")
        f.write(f"Utterance: {' '.join(example.src_sent)}\n")
        f.write(f"Snippet: {example.tgt_code}\n")
        f.write(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    f.close()
    print('Skipped due to exceptions: %d' % len(skipped_list), file=sys.stderr)
    return examples


def preprocess_example(example_json,args=None):
    intent = example_json['intent']
    # intent="how to set global const variables in python?"
    if 'rewritten_intent' in example_json:
        rewritten_intent = example_json['rewritten_intent']
    else:
        rewritten_intent = None

    if rewritten_intent is None:
        rewritten_intent = intent
    snippet = example_json['snippet']

    canonical_intent, slot_map = canonicalize_intent(rewritten_intent)
    
    print("the orig intent is", canonical_intent)
    parsed_intent,_,_=parse_string(core_parser,canonical_intent,False)
    parsed_intent=parsed_intent[0]
    sample=convert_ln(parsed_intent,args=args)


    canonical_snippet = canonicalize_code(snippet, slot_map)
    intent_tokens = tokenize_intent(canonical_intent)
    decanonical_snippet = decanonicalize_code(canonical_snippet, slot_map)

    reconstructed_snippet = astor.to_source(ast.parse(snippet)).strip()
    reconstructed_decanonical_snippet = astor.to_source(ast.parse(decanonical_snippet)).strip()

    assert compare_ast(ast.parse(reconstructed_snippet), ast.parse(reconstructed_decanonical_snippet))

    return {'canonical_intent': canonical_intent,
            'intent_tokens': intent_tokens,
            'leaves_nodes':sample,
            'slot_map': slot_map,
            'canonical_snippet': canonical_snippet}

ntok_counter = Counter()
replaced = Counter()

def convert_ln(sent,args):
    import_user_module(args)
    task = tasks.get_task(args.task)
    src_dict = task.load_dictionary('data/conala/final_vocab.txt')
   
    def replaced_consumer(word, idx):
    # global stat
        if idx == src_dict.unk_index and word != src_dict.unk_word:
            replaced.update([word])
        ntok_counter.update([word])
    """you might just want to use line2example"""
    ex=NstackTreeTokenizer.line2example(s=sent,vocab=src_dict,consumer=replaced_consumer,tokenize=tokenize_line,append_eos=False,
                 reverse_order=False, add_if_not_exist=False, offset=0,end=1,remove_root=True, take_pos_tag=True,
                 take_nodes=True, no_collapse=False, label_only=False, tolower=False)
    sample={k:v for k, v in ex.items()}
    
    return sample

if __name__ == '__main__':
    parser = options.get_preprocessing_parser()
    group = parser.add_argument_group('Preprocessing')
    #### General configuration ####
    group = parser.add_argument_group('Preprocessing')
    group.add_argument("--no_remove_root", action="store_true", help="no_remove_root")
    group.add_argument("--no_take_pos_tag", action="store_true", help="no_take_pos_tag")
    group.add_argument("--no_take_nodes", action="store_true", help="no_take_nodes")
    group.add_argument("--no_reverse_node", action="store_true", help="no_reverse_node")
    group.add_argument("--no_collapse", action="store_true", help="no_collapse")
    args = parser.parse_args()

    # the json files can be downloaded from http://conala-corpus.github.io
    preprocess_conala_dataset(train_file='data/conala/conala-train.json',
                              test_file='data/conala/conala-test.json',
                              mined_data_file=None,
                              api_data_file=None,
                              grammar_file='asdl/lang/py3/py3_asdl.simplified.txt',
                              src_freq=2, code_freq=2,
                              vocab_size=5000,
                              num_mined=0,
                              out_dir='data/conala',args=args)

