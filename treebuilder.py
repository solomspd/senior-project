
import  re
from anytree.render import AsciiStyle
from anytree import Node, RenderTree
from optparse import OptionParser
import sys
import os


address_mapper = {}



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
            if re.findall(".+({.*}|\(.*\));",line):
                functionName = re.findall("\s(\w*?)\(",line)
                args = []
                instructions.clear()
                if('(' in line):
                    args = line[line.find('(')+1: line.find(')')].split(',')
                if(functionName):
                    functionNode = Node(functionName[0], parent= classes[-1])
                type = re.findall("(public|private|protected)",line)
                if type:
                    typeNode = Node(type[0], parent=functionNode)
                else:
                    typeNode = Node("None", parent=functionNode)
                if (functionName): 
                    returnType = re.findall("(?<=\s)(.*?)(?=\s{1,}%s)"%functionName,line)
                    if returnType:
                        returnNode = Node(returnType[0], parent=functionNode)
                else:
                    returnNode = Node("None", parent=functionNode)
                
                for arg in args:
                    if arg:
                        argNode = Node(arg.strip(), parent=functionNode)
                functions.append(functionNode)

            #instructions

            elif re.findall("\d+[:]\s\w+",line):
                instructionInfo = re.findall("[^\s\\:\\\\\/\/<>.\'\"(),;]\w{0,}",line)
                instructionName = instructionInfo[1]
                if(instructions):
                    instructionNode = Node(instructionName, parent=instructions[-1])
                    instructions.append(instructionNode)
                else:
                    instructionNode = Node(instructionName, parent=functions[-1])
                    instructions.append(instructionNode)
                if(len(instructionInfo) > 2):
                    if(re.findall("#{0,1}\d+", instructionInfo[2])):
                        argNode = Node(instructionInfo[2], parent = instructionNode)
                        if(instructionInfo[2][0] == '#'):
                            comment = re.findall("(?<=\/\/ ).*[^;]", line)[0]
                            address_mapper[instructionInfo[2]] = comment
                if(len(instructionInfo) > 3):
                    if(re.findall("#{0,1}\d+", instructionInfo[3])):
                        argNode = Node(instructionInfo[3], parent = instructionNode)
   # for pre, _, node in RenderTree(root):
    #    print("%s%s"%(pre, node.name))



    print(root.children)

            
    return root
    
    






if __name__ == "__main__":
    lines = []
    """
    usage = \""" usage: %prog -f <path> 
    -d: to disassemble the file
    -p: to print the tree 
    \"""

    parser = OptionParser(usage)
    parser.add_option('-f', dest = 'path', type="string", help = "specify the path of the file")
    parser.add_option('-d', dest = "disassemble", action='store_true', help = "specify if you want to disassemble the file")
    parser.add_option('-p', dest = "print", action='store_true', help = "specify if you want to print the tree")

    options, args = parser.parse_args()

    if not options.path:
        print(parser.usage)
        exit(0)
    else:
        root = ""

        with open("prime") as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            root = parse(lines)
        
        if options.disassemble:
            os.system("javap -c " +options.path+"-o ")
    """

    with open ("prime.txt") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
       
    
    root = parse(lines)




        
import  re
from anytree.render import AsciiStyle
from anytree import Node, RenderTree
from optparse import OptionParser
import sys
import os


address_mapper = {}



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
            if re.findall(".+({.*}|\(.*\));",line):
                functionName = re.findall("\s(\w*?)\(",line)
                args = []
                instructions.clear()
                if('(' in line):
                    args = line[line.find('(')+1: line.find(')')].split(',')
                if(functionName):
                    functionNode = Node(functionName[0], parent= classes[-1])
                type = re.findall("(public|private|protected)",line)
                if type:
                    typeNode = Node(type[0], parent=functionNode)
                else:
                    typeNode = Node("None", parent=functionNode)
                if (functionName): 
                    returnType = re.findall("(?<=\s)(.*?)(?=\s{1,}%s)"%functionName,line)
                    if returnType:
                        returnNode = Node(returnType[0], parent=functionNode)
                else:
                    returnNode = Node("None", parent=functionNode)
                
                for arg in args:
                    if arg:
                        argNode = Node(arg.strip(), parent=functionNode)
                functions.append(functionNode)

            #instructions

            elif re.findall("\d+[:]\s\w+",line):
                instructionInfo = re.findall("[^\s\\:\\\\\/\/<>.\'\"(),;]\w{0,}",line)
                instructionName = instructionInfo[1]
                if(instructions):
                    instructionNode = Node(instructionName, parent=instructions[-1])
                    instructions.append(instructionNode)
                else:
                    instructionNode = Node(instructionName, parent=functions[-1])
                    instructions.append(instructionNode)
                if(len(instructionInfo) > 2):
                    if(re.findall("#{0,1}\d+", instructionInfo[2])):
                        argNode = Node(instructionInfo[2], parent = instructionNode)
                        if(instructionInfo[2][0] == '#'):
                            comment = re.findall("(?<=\/\/ ).*[^;]", line)[0]
                            address_mapper[instructionInfo[2]] = comment
                if(len(instructionInfo) > 3):
                    if(re.findall("#{0,1}\d+", instructionInfo[3])):
                        argNode = Node(instructionInfo[3], parent = instructionNode)
   # for pre, _, node in RenderTree(root):
    #    print("%s%s"%(pre, node.name))



    print(root.children)

            
    return root
    
    






if __name__ == "__main__":
    lines = []
    """
    usage = \""" usage: %prog -f <path> 
    -d: to disassemble the file
    -p: to print the tree 
    \"""

    parser = OptionParser(usage)
    parser.add_option('-f', dest = 'path', type="string", help = "specify the path of the file")
    parser.add_option('-d', dest = "disassemble", action='store_true', help = "specify if you want to disassemble the file")
    parser.add_option('-p', dest = "print", action='store_true', help = "specify if you want to print the tree")

    options, args = parser.parse_args()

    if not options.path:
        print(parser.usage)
        exit(0)
    else:
        root = ""

        with open("prime") as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            root = parse(lines)
        
        if options.disassemble:
            os.system("javap -c " +options.path+"-o ")
    """

    with open ("prime.txt") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
       
    
    root = parse(lines)




        
