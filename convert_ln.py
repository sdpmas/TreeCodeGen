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
def run_args():
    parser = options.get_preprocessing_parser()
    group = parser.add_argument_group('Preprocessing')
    group.add_argument("--no_remove_root", action="store_true", help="no_remove_root")
    group.add_argument("--no_take_pos_tag", action="store_true", help="no_take_pos_tag")
    group.add_argument("--no_take_nodes", action="store_true", help="no_take_nodes")
    group.add_argument("--no_reverse_node", action="store_true", help="no_reverse_node")
    group.add_argument("--no_collapse", action="store_true", help="no_collapse")
    args = parser.parse_args()
    get_vocab(args)



if __name__=='__main__':
    run_args()