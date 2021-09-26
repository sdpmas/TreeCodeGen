# coding=utf-8
import argparse


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arg_parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    arg_parser.add_argument('--lang', choices=['python', 'lambda_dcs', 'prolog', 'python3'], default='python',
                            help='[Deprecated] language to parse. Deprecated, use --transition_system and --parser instead')
    arg_parser.add_argument('--asdl_file', type=str, help='Path to ASDL grammar specification')
    arg_parser.add_argument('--mode', choices=['train', 'test', 'interactive',
                                               'train_paraphrase_identifier', 'train_reconstructor','rerank'], required=True, help='Run mode')

    #### Modularized configuration ####
    arg_parser.add_argument('--parser', type=str, default='default_parser', required=False, help='name of parser class to load')
    arg_parser.add_argument('--transition_system', type=str, default='python2', required=False, help='name of transition system to use')
    arg_parser.add_argument('--evaluator', type=str, default='default_evaluator', required=False, help='name of evaluator class to use')

    # Embedding sizes
    arg_parser.add_argument('--embed_size', default=128, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')
    arg_parser.add_argument('--field_embed_size', default=64, type=int, help='Embedding size of ASDL fields')
    arg_parser.add_argument('--type_embed_size', default=64, type=int, help='Embeddings ASDL types')

    # Hidden sizes
    arg_parser.add_argument('--hidden_size', default=256, type=int, help='Size of LSTM hidden states')
    # readout layer
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'],
                            help='Type of activation if using additional linear layer')
    

    #### Training ####
    arg_parser.add_argument('--vocab', type=str, help='Path of the serialized vocabulary')
    arg_parser.add_argument('--train_file', type=str, help='path to the training target file')
    arg_parser.add_argument('--dev_file', type=str, help='path to the dev source file')

    arg_parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    arg_parser.add_argument('--dropout', default=0., type=float, help='Dropout rate')

    # training schedule details
    arg_parser.add_argument('--valid_metric', default='acc', choices=['acc'],
                            help='Metric used for validation')
    arg_parser.add_argument('--log_every', default=10, type=int, help='Log training statistics every n iterations')

    arg_parser.add_argument('--save_to', default='model', type=str, help='Save trained model to')
    arg_parser.add_argument('--patience', default=5, type=int, help='Training patience')
    arg_parser.add_argument('--max_num_trial', default=10, type=int, help='Stop training after x number of trials')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='Maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    arg_parser.add_argument('--lr_decay', default=0.5, type=float,
                            help='decay learning rate if the validation performance drops')
    arg_parser.add_argument('--lr_decay_after_epoch', default=0, type=int, help='Decay learning rate after x epoch')
    arg_parser.add_argument('--decay_lr_every_epoch', action='store_true', default=False, help='force to decay learning rate after each epoch')
    arg_parser.add_argument('--verbose', action='store_true', default=False, help='Verbose mode')

    #### decoding/validation/testing ####
    arg_parser.add_argument('--load_model', default=None, type=str, help='Load a pre-trained model')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--decode_max_time_step', default=100, type=int, help='Maximum number of time steps used '
                                                                                  'in decoding and sampling')
    arg_parser.add_argument('--test_file', type=str, help='Path to the test file')
    arg_parser.add_argument('--save_decode_to', default=None, type=str, help='Save decoding results to file')

    #new one
    arg_parser.add_argument('--workers', default=0, type=int)

    return arg_parser


def update_args(args, arg_parser):
    for action in arg_parser._actions:
        if isinstance(action, argparse._StoreAction) or isinstance(action, argparse._StoreTrueAction) \
                or isinstance(action, argparse._StoreFalseAction):
            if not hasattr(args, action.dest):
                setattr(args, action.dest, action.default)
