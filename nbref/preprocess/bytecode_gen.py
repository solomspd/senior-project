def main():
  parser = argparse.ArgumentParser()
  data_assembly_path = ""
  parser.add_argument('-a', '--asm',
                      help='input assembly folder',
                      required=False, type=str, default=data_assembly_path)
  parser.add_argument('-n', '--num',
                      help='input assembly folder',
                      required=False, type=int, default=10000)
  parser.add_argument('-f', '--feature',
                      help='input feature file',
                      required=False, type=str)
  parser.add_argument('-t', '--task',
                      help='task, branch prediction or prefetching',
                      required=False, type=str, default='pf')
  parser.add_argument('-b', '--binary',
                      help='binary feature',
                      action='store_true', default=False)
  parser.add_argument('-hid', '--hidden_dim',
                      help='hidden dimension',
                      required=False, type=int, default=416)
  args = parser.parse_args()

  graphs = []
  feats  = []
  featuremap = {}
  max_edges = 0
  max_nodes = 0
  token = 1
  
  in_folder  = os.path.join(args.asm, 'rand_assembly') #where
  out_folder = os.path.join(args.asm, 'golden_obj')

  if not os.path.exists(out_folder):
    os.makedirs(out_folder) #create golden obj 

  for nn in range(0,args.num):
    path_ = os.path.join(in_folder,'rd_'+str(nn)+'.s')
    gb = Graph_builder(path_, args.feature, args.task)

    subset = False

    print('==============Nodes=============')
    selected_nodes = []
    for nodes in gb.nodes:
        selected_nodes.append(nodes)

    # renaming node
    if subset:
      new_id2node = {}
      rename = {}
      for i in range(len(selected_nodes)):
        for j in range(len(selected_nodes[i])):
          if selected_nodes[i][j] not in rename:
            rename[selected_nodes[i][j].id] = len(rename)
            selected_nodes[i][j].id = rename[selected_nodes[i][j].id]
            # get new id to node mapping, for later use of getting node name
            # does not overwrite previous
            new_id2node[selected_nodes[i][j].id] = selected_nodes[i][j]
      # give it to gb to pass in dynamic function
      gb.new_id2node = new_id2node
    print(selected_nodes)

    print('==============Edges=============')
    selected_edges = []
    for edge in gb.edges:
      src = gb.id2node[edge.src]
      tgt = gb.id2node[edge.tgt]
      selected_edges.append(edge)

    # renaming edge
    if subset:
      edge_rename = {}
      for i in range(len(selected_edges)):
        selected_edges[i].src = rename[selected_edges[i].src]
        selected_edges[i].tgt = rename[selected_edges[i].tgt]
        if selected_edges[i].type not in edge_rename:
          edge_rename[selected_edges[i].type] = len(edge_rename)
        selected_edges[i].type = edge_rename[selected_edges[i].type]
    print(selected_edges)
    graph = [edge.output() for edge in gb.edges]
    graph.sort(key = operator.itemgetter(1))

    feat = []
    feat_aux = []

    id2idx = {}
    idx = 0
    for elem in gb.nodes:
      for node in elem:
          if isInt(node.name):
              node.name = abs(int(node.name))
          if node.name not in featuremap.keys():
               featuremap[node.name] = token
               token +=1
          if 'tgt' in node.type or 'src' in node.type:
            if 'tgt' in node.type:
              feat_aux.append(1)
            elif 'src' in node.type:
              feat_aux.append(2)
          else:
            feat_aux.append(0)

          feat.append(featuremap[node.name])
          id2idx[node.id] = idx
          idx+=1

    for i, elem in enumerate(graph):
      graph[i] = (id2idx[elem[0]],elem[1],id2idx[elem[2]])

    if max_edges < len(graph):
        max_edges = len(graph)
    if max_nodes < len(feat):
        max_nodes = len(feat)
    graphs.append(graph)
    feats.append(feat)

  print('graph shape in [num_graphs, num_edges, 3]: ', np.shape(graphs))
  print('feature shape in [num_graphs, num_nodes, 64]: ', np.shape(feats))

  np.save(os.path.join(out_folder, 'graphs-3'), graphs)
  np.save(os.path.join(out_folder, 'feats-3'), feats)
