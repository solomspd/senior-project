
import pdb, os, re
from anytree.render import AsciiStyle
import numpy as np
import argparse
import json
from anytree import Node, RenderTree



def parse(lines):
    root = Node("whatever")
    classes = []
    instructions = []
    functions = []
    for line in lines:
        if line:
        #class
            if re.findall("^class", line):
                className = re.findall("\b(?:class )\b|(\w+)", line)[1]
                classNode = Node(className, parent = root)
                classes.append(classNode)
            #Functions
            elif line[-1] == ';':
                functionName = ""
                args = []
                instructions.clear()
                parenthesisIdx = line.find('(')
                if parenthesisIdx:
                    functionName = line[line.rfind(' '):parenthesisIdx]
                    args = line[line.find('(')+1: line.find(')')].split(',')

                
                functionNode = Node(functionName, parent= classes[-1])
                for arg in args:
                    if arg:
                        argNode = Node(arg, parent=functionNode)
                functions.append(functionNode)


           
            #instructions

            elif re.findall("\d+[:]\w*",line):
                instructionInfo = re.findall("\w*",line)
                instructionName = instructionInfo[3]
                if(instructions):
                    instructionNode = Node(instructionName, parent=instructions[-1])
                    instructions.append(instructionNode)
                else:
                    instructionNode = Node(instructionName, parent=functions[-1])
                    instructions.append(instructionNode)
                if(len(instructionInfo) > 7): 
                    if(instructionInfo[7]):
                        argNode = Node(instructionInfo[7], parent = instructionNode)
                
                if(len(instructionInfo) > 13):
                    if(instructionInfo[13]):
                        argNode = Node(instructionInfo[13], parent = instructionNode)
    
    
    for pre, _, node in RenderTree(root):
        print("%s%s"%(pre, node.name))


    

    

                

            


if __name__ == "__main__":
    lines = []
    with open("prime") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

    parse(lines)




