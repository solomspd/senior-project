def generateCode(ast):
    type_map_code = []
    code = []
    type_map_genCode = list(nx.dfs_preorder_nodes(ast, source=0))
    return code 





		
	