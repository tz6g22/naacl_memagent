"""Source-bound expression proposals, evaluated with restricted decimal arithmetic."""
from __future__ import annotations
import ast,re
from decimal import Decimal,InvalidOperation
from typing import Any,Mapping

FORMULA_SCHEMA={'type':'object','additionalProperties':False,'properties':{
 'status':{'type':'string','enum':['COMPUTABLE','INSUFFICIENT_INPUT']},
 'expression':{'type':['string','null'],'maxLength':120},
 'input_refs':{'type':'array','items':{'type':'string'}},
 'reason':{'type':'string','maxLength':160}},
 'required':['status','expression','input_refs','reason']}
FORMULA_SYSTEM='''Propose recomputation from the source derivation and CURRENT numeric input aliases, preserving the target property and units. Never reuse an invalid old input or invent missing inputs.
INPUT: target, current_numeric_inputs.
OUTPUT: status COMPUTABLE|INSUFFICIENT_INPUT, expression, input_refs, reason. Insufficient source basis or inputs: INSUFFICIENT_INPUT and null expression. List each used input alias once.
Use restricted decimal arithmetic: + - * / and parentheses, no calls or attribute access. Constants require source coefficients or ordinary unit/calendar conversion factors. The runtime computes the value; return schema JSON, not an invented numeric answer.'''


def numeric_inputs(nodes:list[Mapping[str,Any]],inverse:Mapping[str,str])->list[dict]:
 result=[]
 for n in nodes:
  # An unambiguous explicit numeric scalar; dates, identifiers and multiple
  # quantities are not silently bound to an arbitrary number.
  value=str(n['value'])
  if re.fullmatch(r'\s*[$£€]?\s*[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|percent)|\s+(?:[A-Za-zµ]+(?:/[A-Za-z]+)?(?:\s+per\s+[A-Za-z]+)?|per\s+[A-Za-z]+))?\s*',value):
   match=re.search(r'[-+]?\d[\d,]*(?:\.\d+)?',value)
   result.append({'ref':inverse[n['state_id']],'entity':n['entity'],'attribute':n['attribute'],
                  'value':value,'unit_text':re.sub(r'^\s*[$£€]?\s*[-+]?\d[\d,]*(?:\.\d+)?','',value).strip(),'number':str(Decimal(match.group().replace(',',''))),
                  'evidence':n['provenance']['text']})
 return result


def calculate(expression:str,bindings:Mapping[str,str])->str:
 tree=ast.parse(expression,mode='eval')
 if len(list(ast.walk(tree)))>64:raise ValueError('expression too complex')
 def visit(n:ast.AST)->Decimal:
  if isinstance(n,ast.Name):
   if n.id not in bindings:raise ValueError('unknown CURRENT numeric input')
   return Decimal(bindings[n.id])
  if isinstance(n,ast.Constant) and type(n.value) in (int,float):return Decimal(str(n.value))
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
   value=visit(n.operand);return value if isinstance(n.op,ast.UAdd) else -value
  if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
   left,right=visit(n.left),visit(n.right)
   if isinstance(n.op,ast.Add):return left+right
   if isinstance(n.op,ast.Sub):return left-right
   if isinstance(n.op,ast.Mult):return left*right
   if not right:raise ValueError('division by zero')
   return left/right
  raise ValueError('unsupported arithmetic syntax')
 try:result=visit(tree.body)
 except InvalidOperation as exc:raise ValueError('invalid decimal arithmetic') from exc
 if not result.is_finite() or abs(result)>Decimal('1e18'):raise ValueError('invalid arithmetic result')
 return format(result.normalize(),'f')


def normalize_formula(raw:Mapping[str,Any],target:Mapping[str,Any],inputs:list[Mapping[str,Any]])->dict:
 refs=raw.get('input_refs');expression=raw.get('expression');by_ref={n['ref']:n for n in inputs}
 if not isinstance(refs,list) or len(refs)!=len(set(refs)) or any(ref not in by_ref for ref in refs):raise ValueError('invalid formula inputs')
 record={'target_state_id':target['state_id'],'target_entity':target['entity'],'target_attribute':target['attribute'],'raw_proposal':dict(raw)}
 if raw.get('status')=='INSUFFICIENT_INPUT':
  if expression is not None:raise ValueError('insufficient formula cannot assert an expression')
  return {**record,'status':'REASSESSMENT_REQUIRED','value':None,'input_refs':refs}
 if raw.get('status')!='COMPUTABLE' or not isinstance(expression,str) or not expression.strip():raise ValueError('invalid formula proposal')
 names={n.id for n in ast.walk(ast.parse(expression,mode='eval')) if isinstance(n,ast.Name)}
 if names!=set(refs) or not names:raise ValueError('formula must use exactly the grounded CURRENT inputs')
 return {**record,'status':'RECOMPUTED','expression':expression,'value':calculate(expression,{r:by_ref[r]['number'] for r in refs}),
         'input_refs':refs,'input_provenance':[dict(by_ref[r]) for r in refs],
         'validation_scope':'CURRENT numeric bindings and exact decimal execution; semantic formula follows persisted source basis'}
