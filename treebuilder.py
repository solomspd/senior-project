
import  re
import dgl
import networkx as nx
import matplotlib.pyplot as plt

address_mapper = {}

def parse(lines):
    data_edge_type = 0
    control_edge_type = 1
    instruction_edge_type = 2
    classes = []
    instructions = []
    functions = []
    graph = nx.DiGraph()
    idx = 0
    for line in lines:
        if line:
            
            #class
            if re.findall("^class", line):
                className = re.findall("\b(?:class )\b|(\w+)", line)[1]
                graph.add_node(idx)
                classes.append(idx)
                idx += 1

            #Functions
            if re.findall(".+({.*}|\(.*\));",line):
                functionName = re.findall("\s(\w*?)\(",line)
                args = []
                instructions.clear()
                if('(' in line):
                    args = line[line.find('(')+1: line.find(')')].split(',')
                if(functionName):
                    graph.add_edge(classes[-1], idx, e_type=control_edge_type)
                    f_idx = idx
                    idx += 1

                type = re.findall("(public|private|protected)",line)
                if type:
                    graph.add_edge(f_idx, idx, e_type=instruction_edge_type)
                    idx += 1

                if (functionName): 
                    returnType = re.findall("(?<=\s)(.*?)(?=\s{1,}%s)"%functionName,line)
                    if returnType:
                        graph.add_edge(f_idx, idx, e_type=instruction_edge_type)
                        idx += 1
                
                for arg in args:
                    if arg:
                        graph.add_edge(f_idx, idx, e_type=instruction_edge_type)
                        idx += 1
                functions.append(f_idx)

            #instructions
            elif re.findall("\d+[:]\s\w+",line):
                instructionInfo = re.findall("[^\s\\:\\\\\/\/<>.\'\"(),;]\w{0,}",line)
                instructionName = instructionInfo[1]
                I_idx = idx
                if(instructions):
                    graph.add_edge(instructions[-1], idx, e_type=control_edge_type)
                    idx += 1
                else:
                    graph.add_edge(functions[-1], idx, e_type=control_edge_type)
                    idx += 1
                if(len(instructionInfo) > 2):
                    if(re.findall("#{0,1}\d+", instructionInfo[2])):
                        graph.add_edge(idx, I_idx, e_type=instruction_edge_type)
                        idx += 1
                        if(instructionInfo[2][0] == '#'):
                            comment = re.findall("(?<=\/\/ ).*[^;]", line)[0]
                            address_mapper[instructionInfo[2]] = comment
                if(len(instructionInfo) > 3):
                    if(re.findall("#{0,1}\d+", instructionInfo[3])):
                        graph.add_edge(idx, I_idx, e_type=instruction_edge_type)
                        idx += 1
                instructions.append(I_idx)
    return dgl.from_networkx(graph, edge_attrs=['e_type'])


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

    with open ("prime") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
       
    
    graph = parse(lines)
    # nx.draw(graph, with_labels=True)
    # plt.show()
    # dgl_g = dgl.from_networkx(graph)

    print(graph)