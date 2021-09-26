from fairseq import tasks,options
from fairseq.data import dictionary
from fairseq.utils import import_user_module
from fairseq.tokenizer import tokenize_line
from tokenizer import * 
from dptree.nstack_process import *
from dptree.tree_builder import *

from dptree.tree_process import *
from binarization import *
from tokenizer import * 
from data.dptree_index_dataset import *
def make_binary_nstack_dataset(vocab, num_workers,args):
    remove_root = not args.no_remove_root
    take_pos_tag = not args.no_take_pos_tag
    take_nodes = not args.no_take_nodes
    reverse_node = not args.no_reverse_node
    no_collapse = args.no_collapse
    print(" Dictionary: {} types".format(len(vocab) - 1))

    input_file='data/conala/parsed_src.txt'

    dss = {
        modality: NstackSeparateIndexedDatasetBuilder(f'data/conala/final_{modality}.bin')
        for modality in NSTACK_KEYS
    }

    def consumer(example):
        for modality, tensor in example.items():
            dss[modality].add_item(tensor)

    stat = NstackTreeMergeBinarizerDataset.export_binarized_separate_dataset(
        input_file, vocab, consumer, add_if_not_exist=False, num_workers=num_workers,
        remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes, reverse_node=reverse_node,
        no_collapse=no_collapse,
    )
    ntok = stat['ntok']
    nseq = stat['nseq']
    nunk = stat['nunk']

    for modality, ds in dss.items():
        ds.finalize(f'data/conala/final_{modality}.idx')

    print(
        "|{}: {} sents, {} tokens, {:.3}% replaced by {}".format(
    
            input_file,
            nseq,
            ntok,
            100 * nunk / ntok,
            vocab.unk_word,
        )
    )
    for modality, ds in dss.items():
        print(f'\t{modality}')

def build_vocab(_src_file,args):
    remove_root = not args.no_remove_root
    take_pos_tag = not args.no_take_pos_tag
    take_nodes = not args.no_take_nodes
    no_collapse = args.no_collapse
    d = dictionary.Dictionary()
    print(f'Build dict on src_file: {_src_file}')
    NstackTreeTokenizer.acquire_vocab_multithread(
        _src_file, d, tokenize_line, num_workers=args.workers,
        remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes,
        no_collapse=no_collapse,
    )
    #TODO: change this to include the target option too.
    d.finalize(
        threshold=args.thresholdsrc,
        nwords=args.nwordssrc,
        padding_factor=args.padding_factor
    )
    print(f'Finish building src vocabulary: size {len(d)}')
    return d

def get_vocab(args):
    import_user_module(args)
    task = tasks.get_task(args.task)
    src_dict=build_vocab("data/conala/parsed_src.txt",args)
    src_dict.save('data/conala/final_vocab.txt')
    make_binary_nstack_dataset(src_dict,args.workers,args)
def run_args():
    parser = options.get_preprocessing_parser()
    group = parser.add_argument_group('Preprocessing')
    group.add_argument("--no_remove_root", action="store_true", help="no_remove_root")
    group.add_argument("--no_take_pos_tag", action="store_true", help="no_take_pos_tag")
    group.add_argument("--no_take_nodes", action="store_true", help="no_take_nodes")
    group.add_argument("--no_reverse_node", action="store_true", help="no_reverse_node")
    group.add_argument("--no_collapse", action="store_true", help="no_collapse")
    args = parser.parse_args()
    args.srcdict='data/conala/src.vocab'
    args.destdir='data/conala'
    args.output_format="binary"
    get_vocab(args)



if __name__=='__main__':
    run_args()