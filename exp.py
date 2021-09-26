# coding=utf-8
from __future__ import print_function
import time
import six.moves.cPickle as pickle
import evaluation
from common.registerable import Registrable
import sys
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from common.utils import init_arg_parser
from model.parser import Parser
from args import *
from components.dataset_new import Dataset
import numpy as np
def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args

def train(args):
    """Maximum Likelihood Estimation"""
    args.cuda = True if torch.cuda.is_available() else False
    args.decay_lr_every_epoch = True
    train_set = Dataset.from_bin_file(args.train_file)
    if args.dev_file:
        dev_set = Dataset.from_bin_file_dev(args.dev_file)
    else:
        dev_set = Dataset(examples=[])
    
    print(len(train_set), len(dev_set),'these are the lengths of train and dev set')
    vocab = pickle.load(open(args.vocab, 'rb'))

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = Registrable.by_name(args.transition_system)(grammar)
    model = Parser.build_model(args, vocab, transition_system)
    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("the total trainable parameters are: ",pytorch_total_params)

    evaluator = Registrable.by_name(args.evaluator)(
        transition_system, args=args)
    if args.cuda:
        model.cuda()

    optimizer_cls = eval('torch.optim.%s' % args.optimizer)
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    pat=0
    while True:
        epoch += 1
        epoch_begin = time.time()
        total_loss=0
        iters=0
        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(
                e.tgt_actions) <= args.decode_max_time_step]
            train_iter += 1
            optimizer.zero_grad()
            ret_val = model(batch_examples)
            loss = -ret_val[0]
            report_loss +=torch.sum(loss).data.item() 
            report_examples += len(batch_examples)
            loss = torch.mean(loss)
            total_loss+=loss
            iters+=1
            loss.backward()
            optimizer.step()
            if train_iter % args.log_every == 0:
                log_str = '[Iter %d] encoder loss=%.5f' % (
                    train_iter, report_loss / report_examples)

                print(log_str)
                report_loss = report_examples = 0.
        print("loss for this epoch: ",total_loss/iters)
        if args.decay_lr_every_epoch and epoch > 10:
            lr = optimizer.param_groups[0]['lr'] * 0.95
            print('decay learning rate to %f' % lr, file=sys.stderr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print('[Epoch %d] epoch elapsed %ds' %
              (epoch, time.time() - epoch_begin), file=sys.stderr)

        if epoch > 20:
            is_better = False
            if args.dev_file:
                print('dev file validation')
                eval_start = time.time()
                eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
                                                   verbose=False, eval_top_pred_only=args.eval_top_pred_only)
              
                dev_score = eval_results[evaluator.default_metric]

                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                    epoch, eval_results,
                    evaluator.default_metric,
                    dev_score,
                    time.time() - eval_start), file=sys.stderr)
                is_better = history_dev_scores == [
                ] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)
                model.train()
            else:
                is_better = True
            if is_better:
                model_file = f'saved_models/dev/{epoch}.bin'
                if history_dev_scores:
                    s = history_dev_scores[-1]
                    model_file = f'saved_models/dev/{epoch}${s}.bin'
                print('save model to [%s]' % model_file)
                model.save(model_file)
        if epoch == args.max_epoch:
            print('max epoch, stopping training')
            exit(0)


def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    args.lang = saved_args.lang
    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    parser.eval()
    evaluator = Registrable.by_name(args.evaluator)(
        transition_system, args=args)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    print(eval_results, file=sys.stderr)


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hier(args)
    print(args)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        print('mode not valid')
        exit(1)
