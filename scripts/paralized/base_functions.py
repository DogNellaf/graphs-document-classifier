def singular_subsumption_check(graph, subgraph):
    intersection = [e for e in subgraph.edges(data=True) if e in graph.edges(data=True)]

    if intersection == list(subgraph.edges(data=True)):
        subsumption = 1

    else:
        subsumption = 0

    return subsumption


def multiple_subsumption_check(graph, subgraphs):
    for subgraph in subgraphs:
        singular_subsumption = singular_subsumption_check(graph, subgraph)

        if singular_subsumption == 0:
            multiple_subsumption = 0
            return multiple_subsumption

    multiple_subsumption = 1
    return multiple_subsumption


def find_graph_pattern_weight(graph_pattern):
    graph_pattern_weight = 0

    for i in range(len(graph_pattern['subgraphs'])):
        graph_pattern_weight += (len(graph_pattern['subgraphs'][i].nodes) * graph_pattern['supports'][i])

    return graph_pattern_weight


def find_size(subgraphs):
    size = 0

    for subgraph in subgraphs:
        size += len(subgraph.nodes)

    return size


def find_maximal_degree(subgraphs):
    maximal_degree = 0

    for subgraph in subgraphs:
        if int(sorted(subgraph.degree, key=lambda x: x[1], reverse=True)[0][1]) > maximal_degree:
            maximal_degree = int(sorted(subgraph.degree, key=lambda x: x[1], reverse=True)[0][1])

    return maximal_degree
