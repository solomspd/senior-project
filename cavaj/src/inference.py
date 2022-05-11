import argparse
import torch
from pathlib import Path

from model.model_top import cavaj
from data_proc import data_proc

import inspect
import javalang
import sys
import javalangSelect
from copy import deepcopy
import networkx as nx
from dotdict import dotdict
import torch_geometric.transforms as T

type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)] 
# print(type_map)

def generateCode(ast, file):
    type_map_code = []
    codeConv = []
    code = []
    type_map_genCode = list(nx.dfs_preorder_nodes(ast, source=0))
    for item in type_map_genCode:
        codeline = code.append(type_map[item])
        code.append(javalangSelect.ret_string(codeline))
    return code 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model inference')
    parser.add_argument('--model', type=str, metavar='checkpoint')
    parser.add_argument('--llc', type=str, metavar='input file', nargs='+')
    arg = parser.parse_args()

    checkpoint = torch.load(arg.model, map_location='cpu')
    model_arg = dotdict(checkpoint['args'])
    model_arg.device = torch.device('cpu')
    model = cavaj(model_arg)
    model.load_state_dict(checkpoint['model_state_dict'])
    data_processor = data_proc()

    instruction_identifier = ['aaload', 'aastore', 'aconst_null', 'aload', 'aload_0', 'aload_1', 'aload_2', 'aload_3', 'anewarray', 'areturn', 'arraylength', 'astore', 'astore_0', 'astore_1', 'astore_2', 'astore_3', 'athrow', 'baload', 'bastore', 'bipush', 'breakpoint', 'caload', 'castore', 'checkcast', 'd2f', 'd2i', 'd2l', 'dadd', 'daload', 'dastore', 'dcmpg', 'dcmpl', 'dconst_0', 'dconst_1', 'ddiv', 'dload', 'dload_0', 'dload_1', 'dload_2', 'dload_3', 'dmul', 'dneg', 'drem', 'dreturn', 'dstore', 'dstore_0', 'dstore_1', 'dstore_2', 'dstore_3', 'dsub', 'dup', 'dup_x1', 'dup_x2', 'dup2', 'dup2_x1', 'dup2_x2', 'f2d', 'f2i', 'f2l', 'fadd', 'faload', 'fastore', 'fcmpg', 'fcmpl', 'fconst_0', 'fconst_1', 'fconst_2', 'fdiv', 'fload', 'fload_0', 'fload_1', 'fload_2', 'fload_3', 'fmul', 'fneg', 'frem', 'freturn', 'fstore', 'fstore_0', 'fstore_1', 'fstore_2', 'fstore_3', 'fsub', 'getfield', 'getstatic', 'goto', 'goto_w', 'i2b', 'i2c', 'i2d', 'i2f', 'i2l', 'i2s', 'iadd', 'iaload', 'iand', 'iastore', 'iconst_m1', 'iconst_0', 'iconst_1', 'iconst_2', 'iconst_3', 'iconst_4', 'iconst_5', 'idiv', 'if_acmpeq', 'if_acmpne', 'if_icmpeq', 'if_icmpge', 'if_icmpgt', 'if_icmple', 'if_icmplt', 'if_icmpne', 'ifeq', 'ifge', 'ifgt', 'ifle', 'iflt', 'ifne', 'ifnonnull', 'ifnull', 'iinc', 'iload', 'iload_0', 'iload_1', 'iload_2', 'iload_3', 'impdep1', 'impdep2', 'imul', 'ineg', 'instanceof', 'invokedynamic', 'invokeinterface', 'invokespecial', 'invokestatic', 'invokevirtual', 'ior', 'irem', 'ireturn', 'ishl', 'ishr', 'istore', 'istore_0', 'istore_1', 'istore_2', 'istore_3', 'isub', 'iushr', 'ixor', 'jsr†', 'jsr_w†', 'l2d', 'l2f', 'l2i', 'ladd', 'laload', 'land', 'lastore', 'lcmp', 'lconst_0', 'lconst_1', 'ldc', 'ldc_w', 'ldc2_w', 'ldiv', 'lload', 'lload_0', 'lload_1', 'lload_2', 'lload_3', 'lmul', 'lneg', 'lookupswitch', 'lor', 'lrem', 'lreturn', 'lshl', 'lshr', 'lstore', 'lstore_0', 'lstore_1', 'lstore_2', 'lstore_3', 'lsub', 'lushr', 'lxor', 'monitorenter', 'monitorexit', 'multianewarray', 'new', 'newarray', 'nop', 'pop', 'pop2', 'putfield', 'putstatic', 'ret†', 'return', 'saload', 'sastore', 'sipush', 'swap', 'tableswitch', 'wide', '(no name)']
    ohd_llc = T.OneHotDegree(len(instruction_identifier), cat=False)

    for i in arg.llc:
        llc_path = Path(i)
        with open (llc_path) as file:
            try:
                llc = data_processor.load_bytecode(file)
            except Exception as e:
                print(f'Failed to load Low Level Code file due to {e}')
                continue
        ohd_llc(llc)
        pred = model(llc)
        print(pred)
        