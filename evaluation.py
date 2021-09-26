# coding=utf-8
from __future__ import print_function

import sys
import traceback
from tqdm import tqdm


def decode(examples, model, args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()
    # c=True if cuda.is_available() else False

    decode_results = []
    count = 0
    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        # print(example.input_actions)
        if not args.cuda:
            hyps = model.parse(example.leaves_nodes['leaves'].unsqueeze(dim=0).long(),example.leaves_nodes['nodes'].unsqueeze(dim=0).long(),example.leaves_nodes['spans'].unsqueeze(dim=0),orig_leaves=example.leaves_nodes['orig_leaves'],context=None, beam_size=args.beam_size)
        else:
            hyps = model.parse(example.leaves_nodes['leaves'].unsqueeze(dim=0).long().cuda(),example.leaves_nodes['nodes'].unsqueeze(dim=0).long().cuda(),example.leaves_nodes['spans'].unsqueeze(dim=0).cuda(),orig_leaves=example.leaves_nodes['orig_leaves'],context=None, beam_size=args.beam_size)


        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            
            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                
                decoded_hyps.append(hyp)
            except:
                print(traceback.format_exc())

        count += 1

        decode_results.append(decoded_hyps)

    if was_training: model.train()

    return decode_results


def evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=False):
    
    examples=examples
    decode_results = decode(examples, parser, args, verbose=verbose)

    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only, args=args,transition_system=parser.transition_system)

    if return_decode_result:
        #TODO: remove this
        return eval_result, decode_results
    else:
        return eval_result
