# coding=utf-8

"""all done"""
class Action(object):
    pass


class ApplyRuleAction(Action):
    def __init__(self, production):
        self.production = production

    def __hash__(self):
        return hash(self.production)

    def __eq__(self, other):
        return isinstance(other, ApplyRuleAction) and self.production == other.production

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'ApplyRule[%s]' % self.production.__repr__()

class GenTextAction(Action):
    def __init__(self, text):
        self.text=text

    def __repr__(self):
        return 'GenText[%s]'%self.text

class MaskAction(Action):
    def __init__(self):
        self.mask='mask'
    def __repr__(self):
        return 'Mask'



class LangMask(Action):
    def __init__(self):
        self.mask='mask'
    def __repr__(self):
        return 'LangMask'
class GenTokenAction(Action):
    def __init__(self, token):
        self.token = token

    def is_stop_signal(self):
        return self.token == '</primitive>'

    def __repr__(self):
        return 'GenToken[%s]' % self.token


class ReduceAction(Action):
   def __repr__(self):
       return 'Reduce'

class TreeTextAction(Action):
    def __init__(self, text):
        self.text=text
    def __repr__(self):
       return 'TreeText[%s]' % self.text


class TransitionSystem(object):
    def __init__(self, grammar):
        self.grammar = grammar

    def get_actions(self, asdl_ast=None,text=None,grammar=None):
        """
        generate action sequence given the ASDL Syntax Tree
        """
        actions=[]
        code_actions = []
        
        if asdl_ast is not None:
            
            parent_action = ApplyRuleAction(asdl_ast.production)
            # print ('production',asdl_ast.production)
            
            code_actions.append(parent_action)
            actions.append(parent_action)
            # text=text.strip().split()
        

        if text is not None:
            text_actions=[]
            if type(text).__name__!='list':
                text=text.strip().split()
            assert type(text).__name__=='list'
           
            for t in text:
                
                text_actions.append(GenTextAction(t))
                # actions.append(ApplyRuleAction(prod))
                # actions.append(ApplyRuleAction(prod_expr))
                # print (ApplyRuleAction('expr -> Name(identifier id)').production.fields)
                actions.append(GenTextAction(t))
            # print (text_actions[:20], 'these are the text actions')
        # else:

            #     text=text.strip().split()
            #     for t in text:
            #         text_actions.append(GenTextAction(t))
            #         actions.append(GenTextAction(t))
       
        if asdl_ast is not None:
            for field in asdl_ast.fields:
                # is a composite field
                if self.grammar.is_composite_type(field.type):
                    if field.cardinality == 'single':
                        field_actions = self.get_actions(asdl_ast=field.value,text=None)
                    else:
                        field_actions = []

                        if field.value is not None:
                            if field.cardinality == 'multiple':
                                for val in field.value:
                                    cur_child_actions = self.get_actions(asdl_ast=val,text=None)
                                    field_actions.extend(cur_child_actions)
                            elif field.cardinality == 'optional':
                                field_actions = self.get_actions(asdl_ast=field.value)

                        # if an optional field is filled, then do not need Reduce action
                        if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                            field_actions.append(ReduceAction())
                else:  # is a primitive field
                    
                    field_actions = self.get_primitive_field_actions(field)
                    # print (field_actions)

                    # if an optional field is filled, then do not need Reduce action
                    if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                        # reduce action
                        field_actions.append(ReduceAction())

                code_actions.extend(field_actions)
                actions.extend(field_actions)
                
        # print (code_actions[:20],"these are the code actions")
        if text is not None and asdl_ast is not None:
            
            assert len([code_actions[0]])+len(text_actions)+len(code_actions[1:])==len(actions)
            
            return actions,text_actions,code_actions
        elif asdl_ast is not None:
            assert actions==code_actions
            return actions
        else:
            assert len(text_actions)==len(actions)
            return actions
    def get_actions1(self, asdl_ast):
        """
        generate action sequence given the ASDL Syntax Tree
        """

        actions = []

        parent_action = ApplyRuleAction(asdl_ast.production)
        actions.append(parent_action)

        for field in asdl_ast.fields:
            # is a composite field
            if self.grammar.is_composite_type(field.type):
                if field.cardinality == 'single':
                    field_actions = self.get_actions(field.value)
                else:
                    field_actions = []

                    if field.value is not None:
                        if field.cardinality == 'multiple':
                            for val in field.value:
                                cur_child_actions = self.get_actions(val)
                                field_actions.extend(cur_child_actions)
                        elif field.cardinality == 'optional':
                            field_actions = self.get_actions(field.value)

                    # if an optional field is filled, then do not need Reduce action
                    if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                        field_actions.append(ReduceAction())
            else:  # is a primitive field
                field_actions = self.get_primitive_field_actions(field)

                # if an optional field is filled, then do not need Reduce action
                if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                    # reduce action
                    field_actions.append(ReduceAction())

            actions.extend(field_actions)

        return actions

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError

    def get_primitive_field_actions(self, realized_field):
        raise NotImplementedError

    #since we are currently experimenting with l2c only, gentext is not a valid continuation type.
    def get_valid_continuation_types(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction,
                else:  # optional, multiple
                    return ApplyRuleAction, ReduceAction
            else:
                if hyp.frontier_field.cardinality == 'single':
                    return GenTokenAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    if hyp._value_buffer:
                        return GenTokenAction,
                    else:
                        """some changes here"""
                        return GenTokenAction, ReduceAction
                else:
                    return GenTokenAction, ReduceAction
        else:
            return ApplyRuleAction,

    def get_valid_continuating_productions(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                return self.grammar[hyp.frontier_field.type]
            else:
                raise ValueError
        else:
            return self.grammar[self.grammar.root_type]

    @staticmethod 
    def get_class_by_lang(lang):
        if lang == 'python':
            from .lang.py.py_transition_system import PythonTransitionSystem
            return PythonTransitionSystem
        elif lang == 'python3':
            from .lang.py3.py3_transition_system import Python3TransitionSystem
            return Python3TransitionSystem
        elif lang == 'lambda_dcs':
            from .lang.lambda_dcs.lambda_dcs_transition_system import LambdaCalculusTransitionSystem
            return LambdaCalculusTransitionSystem
        elif lang == 'prolog':
            from .lang.prolog.prolog_transition_system import PrologTransitionSystem
            return PrologTransitionSystem

        raise ValueError('unknown language %s' % lang)
