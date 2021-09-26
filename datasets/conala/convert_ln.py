from fairseq import tasks,options
from fairseq import dictionary
from fairseq.utils import import_user_module


def get_vocab(args):
    import_user_module(args)
    task = tasks.get_task(args.task)
    src_dict = task.load_dictionary(args.srcdict)
    src_dict.save("data/conala/src_dict.txt")


def run_args():
    parser = options.get_preprocessing_parser()
    group = parser.add_argument_group('Preprocessing')
    args = parser.parse_args()
    args.srcdict='data/conala/src.vocab'
    args.output_format="binary"
    get_vocab(args)



if __name__=='__main__':
    run_args()