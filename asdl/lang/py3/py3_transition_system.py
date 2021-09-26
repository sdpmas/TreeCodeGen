# coding=utf-8

import ast

import astor

from asdl.lang.py.py_asdl_helper import asdl_ast_to_python_ast, python_ast_to_asdl_ast
from asdl.lang.py.py_utils import tokenize_code
from asdl.transition_system import TransitionSystem, GenTokenAction,GenTextAction

from common.registerable import Registrable


@Registrable.register('python3')
class Python3TransitionSystem(TransitionSystem):
    def tokenize_code(self, code, mode=None):
        return tokenize_code(code, mode)

   

    def surface_code_and_text_to_ast(self, code, text):
        py_code_ast=ast.parse(code)
        return python_ast_to_asdl_ast(py_code_ast, text, self.grammar)

    def ast_to_surface_code(self, asdl_ast):
        py_ast = asdl_ast_to_python_ast(asdl_ast, self.grammar)
        code = astor.to_source(py_ast).strip()

        if code.endswith(':'):
            code += ' pass'

       
        return code
    
    def compare_ast(self, hyp_ast, ref_ast):
        hyp_code,hyp_text = self.ast_to_surface_code_and_text(hyp_ast)
        ref_code,ref_text = self.ast_to_surface_code_and_text(ref_ast)

        ref_code_tokens = tokenize_code(ref_code)
        hyp_code_tokens = tokenize_code(hyp_code)
        # hyp_text=hyp_text.strip().split()
        # ref_text=ref_text.strip().split()

        return ref_code_tokens == hyp_code_tokens and hyp_text==ref_text

    def get_primitive_field_actions_old(self, realized_field):
        actions = []
        if realized_field.value is not None:
            if realized_field.cardinality == 'multiple':  # expr -> Global(identifier* names)
                field_values = realized_field.value
            else:
                field_values = [realized_field.value]

            # token_text=[]
            token=[]
            if realized_field.type.name == 'string':
                for field_val in field_values:

                    tokens_st.extend(field_val.split(' ') + ['</primitive>'])
            else:
                for field_val in field_values:
                    tokens.append(field_val)

            for tok in tokens:
               
                if realized_field.type=='String':
                    actions.append(GenTokenAction(tok))
                else:
                    actions.append(GenTextAction(tok))
    
                
        elif realized_field.type.name == 'singleton' and realized_field.value is None:
            # singleton can be None
            if realized_field.type=='String':
                actions.append(GenTokenAction('None'))
            else:
                actions.append(GenTextAction('None'))
            # actions.append(GenTokenAction('None'))

        return actions

    def get_primitive_field_actions(self, realized_field):
        actions = []
        if realized_field.value is not None:
            if realized_field.cardinality == 'multiple':  # expr -> Global(identifier* names)
                field_values = realized_field.value
            else:
                field_values = [realized_field.value]

            tokens = []
            if realized_field.type.name == 'string':
                for field_val in field_values:
                    tokens.extend(field_val.split(' ') + ['</primitive>'])
            else:
                for field_val in field_values:
                    tokens.append(field_val)

            for tok in tokens:
                actions.append(GenTokenAction(tok))
        elif realized_field.type.name == 'singleton' and realized_field.value is None:
            # singleton can be None
            actions.append(GenTokenAction('None'))

        return actions
        
    def is_valid_hypothesis(self, hyp, **kwargs):
        try:
            hyp_code = self.ast_to_surface_code(hyp.tree)
            ast.parse(hyp_code)
            self.tokenize_code(hyp_code)
        except:
            return False
        return True
