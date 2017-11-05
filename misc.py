import joblib


def savez(obj, filename, protocol=0):
    """Saves a compressed object to disk

    """
    joblib.dump(obj, filename, compress=True)


def loadz(filename):
    """Loads a compressed object from disk
    """
    return joblib.load(filename)


def dfs_topsort(graph, root=None):
    """ Perform topological sort using a modified depth-first search

    Parameters
    ----------
    graph: dict
        A directed graph

    root: str, int or None
        We perform topological sort for a subgraph of the given node.
        If `root` is None, the entire graph is considered.


    Return
    ------
    L: list
        a sorted list of nodes in the input graph

    """

    L = []
    color = {u: "white" for u in graph}
    found_cycle = [False]

    subgraph = graph
    if root is not None:
        try:
            subgraph = graph[root]
        except KeyError as e:
            raise(e)

    for u in subgraph:
        if color[u] == "white":
            dfs_visit(graph, u, color, L, found_cycle)
        if found_cycle[0]:
            break

    if found_cycle[0]:
        L = []

    L.reverse()
    return L


def dfs_visit(graph, u, color, L, found_cycle):
    if found_cycle[0]:
        return

    color[u] = "gray"
    for v in graph[u]:
        if color[v] == "gray":
            print('Found cycle by {}'.format(u))
            found_cycle[0] = True
            return
        if color[v] == "white":
            dfs_visit(graph, v, color, L, found_cycle)
    color[u] = "black"
    L.append(u)
