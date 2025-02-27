#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:01:21 2020

@author: A.Goumilevski
"""

import re, os

path = os.path.dirname(os.path.abspath(__file__))

def createGraph(model,img_file_name="Equations_Graph.png"):
    """
    Create a graph of endogenous variables of a model.
    
    Parameters:
        :param model: The Model object.
        :type model: Instance of class Model.
    """
    import pydot
    from snowdrop.src.utils.util import findVariableLead,findVariableLag
    from snowdrop.src.grahs.graph import Graph
        
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")"
    regexPattern = '|'.join(map(re.escape, delimiters))
      
    endog = model.symbols["variables"]
    eqs   = model.symbolic.equations
    n_eqs = len(eqs) - model.symbolic.numberOfNewEqs
    lst   = list(); lst1 = list()
    m     = dict(); m1 = dict(); m2 = dict(); m3 = dict()
            
    graph = pydot.Dot(graph_type='digraph')
    graph.set_size('"50,10!"')
   
    # Find node colors
    for i in range(n_eqs):
        eqtn = eqs[i].replace("(+1)","(1)")
        if "=" in eqtn:
            # Left hand side of equation
            i = eqtn.index("=")
            eq = eqtn[:i].replace(" ","")
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            # Build nodes for left-side-side variables   
            for v in arr:
                if v in endog:
                    ind = eq.find(v)
                    eq = eq[ind+len(v):]
                    if eq.startswith("("):
                        ind = eq.index(")")
                        v1 = v + eq[:ind+1]
                        eq = eq[ind:]
                    else:
                        v1 = v
                    if not v1 in lst:
                        lst.append(v1)
            # Right-hand-side of equation
            eq = eqtn[1+i:].replace(" ","")
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            for v in arr:
                if v in endog:
                    ind = eq.find(v)
                    eq = eq[ind+len(v):]
                    if eq.startswith("("):
                        ind = eq.index(")")
                        v1 = v + eq[:ind+1]
                        eq = eq[ind:]
                    else:
                        v1 = v
                    if v1 in lst:
                        lst.remove(v1)
                    
    edges = []
    for i in range(n_eqs):
        eqtn = eqs[i].replace("(1)","(+1)")
        if "=" in eqtn:
            # Left side of equation
            ind = eqtn.index("=")
            eq = eqtn[:ind].replace(" ","")
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            lst1 = list()
            # Build nodes for the left-hand-side variables   
            for v in arr:
                if v in endog:
                    ind1 = eq.find(v)
                    eq = eq[ind1+len(v):]
                    if eq.startswith("("):
                        ind = eq.index(")")
                        v1 = v + eq[:ind1+1]
                        eq = eq[ind1:]
                    else:
                        v1 = v
                    if not v1 in lst1:
                        lst1.append(v1)
            # Right hand side of equation
            eq = eqtn[1+ind:].replace(" ","")
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            lst2 = list()
            for v in arr:
                if v in endog:
                    ind2 = eq.find(v)
                    eq = eq[ind2+len(v):]
                    if eq.startswith("("):
                        ind2 = eq.index(")")
                        v1 = v + eq[:ind2+1]
                        eq = eq[ind2:]
                    else:
                        v1 = v
                    if not v1 in lst2:
                        lst2.append(v1)
                    
                    
            # Map nodes of the left-hand-side variables to the righ-hand-side variables 
            for name in lst1: 
                if not name in m:
                    m3[name] = []
                l = m3[name]
                for name2 in lst2:
                    if not name2 in l:
                        l +=  [name2] 
                    # Make map symmetric    
                    if not name2 in m:
                        m3[name2] = []
                        
        # Build nodes for the left-hand-side variables 
        for v in lst1:
            # Add nodes to the graph
            if v in m:
                left_node = m[v]
            else:
                name = v
                lead = findVariableLead(v)
                lag  = findVariableLag(v)
                if lead > 0 and "_p" in v:
                    ind = v.index("_p")
                    name = v[:ind] + "(" + str(lead) + ")"
                elif lag < 0 and "_m" in v:
                    ind = v.index("_m")
                    name = v[:ind] + "(" + str(lag) + ")"
                color = "green" if v in lst else "yellow"
                left_node = pydot.Node(name,style="filled",fillcolor=color,fontsize=max(15,n_eqs))
                graph.add_node(left_node)
                m[v] = left_node
                m2[name] = left_node
            # Build nodes of right-hand-side equation variables   
            for v2 in lst2:
                # Add the nodes to the graph
                if v2 in m:
                    right_node = m[v2]
                    name2 = right_node.get_name()
                    name2 = name2.replace('"','')
                else:
                    name2 = v2 
                    lead = findVariableLead(v2)
                    lag  = findVariableLag(v2)
                    if lead > 0 and "_p" in v2:
                        ind = v2.index("_p")
                        name2 = v2[:ind] + "(" + str(lead) + ")"
                    elif lag < 0 and "_m" in v2:
                        ind = v2.index("_m")
                        name2 = v2[:ind] + "(" + str(lag) + ")"
                    right_node = pydot.Node(name2,style="filled",fillcolor="yellow",fontsize=max(15,n_eqs))
                    graph.add_node(right_node)
                    m[v2] = right_node
                    m2[name2] = right_node
                # Create the edges   
                if not v == v2 and not name == name2:
                    k = left_node.get_name().replace('"','')+"->"+right_node.get_name().replace('"','')
                    if k in edges:
                        pass
                    else:
                        #print(k)
                        graph.add_edge(pydot.Edge(left_node, right_node))
                    edges.append(k)
                    if not name in m1:
                        m1[name] = list()
                    m1[name].append(name2)
                        
                  
    # # Fix missing egdes
    # for k in m2:
    #     n = k; n2 = None
    #     if n.endswith("(+1)"):
    #         n2 = n[:-3]
    #     elif n.endswith("(+1)"):
    #         n2 = n
    #         n = n[:-3]
        
    #     if n in m2 and n2 in m2:
    #         node1 = m2[n]
    #         node2 = m2[n2]
    #         if not n in m1 or not n2 in m1[n]:
    #             graph.add_edge(pydot.Edge(node1, node2))
                        
                        
    fpath = os.path.join(path,'../../graphs',img_file_name)
    graph.write_png(fpath)
    
    # Build graph object
    G = Graph(m3)
    print(G)
    

def createClusters(model,img_file_name="Minimum_Spanning_Tree.png"):
    """
    Create graph of components of model equations endogenous variables.
    
    Parameters:
        :param model: The Model object.
        :type model: Instance of class Model.
    """
    import networkx as nx
        
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")"
    regexPattern = '|'.join(map(re.escape, delimiters))
      
    endog = model.symbols["variables"]
    eqs   = model.symbolic.equations
    n_eqs = len(eqs) - model.symbolic.numberOfNewEqs
    m     = dict()
    nodes = set()
    lst   = list()    
    
    for i in range(n_eqs):
        eqtn = eqs[i].replace("(+1)","(1)")
        if "=" in eqtn:
            # Left side of equation
            i = eqtn.index("=")
            eq = eqtn[:i].replace(" ","")
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            lst1 = list()
            # Build nodes for the left-hand-side variables   
            for v in arr:
                if v in endog:
                    name = v
                    if "_p" in v:
                        ind = v.index("_p")
                        name = v[:ind]
                    elif "_m" in v:
                        ind = v.index("_m")
                        name = v[:ind]
                    if not name in lst1:
                        lst1.append(name)
                    if not name in lst:
                        lst.append(name)
            # Right hand side of equation
            eq = eqtn[1+i:].replace(" ","")
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            lst2 = list()
            for v in arr:
                if v in endog:
                    name = v
                    if "_p" in v:
                        ind = v.index("_p")
                        name = v[:ind]
                    elif "_m" in v:
                        ind = v.index("_m")
                        name = v[:ind]
                    if not name in lst2:
                        lst2.append(name)
                    if name in lst:
                        lst.remove(name)
                    
            # Map nodes for the left-hand-side variables to the righ-hand-side variables 
            for name in lst1: 
                if not name in m:
                    m[name] = []
                    nodes.add(name)
                l = m[name]
                for name2 in lst2:
                    if not name2 in l:
                        l +=  [name2] 
                        nodes.add(name2)
                    # Make map symmetric    
                    if not name2 in m:
                        m[name2] = []
                    l2 = m[name2]
                    if not name in l2:
                        l2 +=  [name] 
    size = len(nodes)                    
                  
    G = nx.Graph()
    G.add_nodes_from(m)
    for k in m:
        nodes = m[k]
        for k2 in nodes:
            G.add_edge(k,k2)
            
    colors = ["blue" if node in lst else "yellow" for node in G.nodes()]
    
    # Extract minimum spanning tree
    G = nx.minimum_spanning_tree(G, weight='length')
    
    fpath = os.path.join(path,'../../graphs',img_file_name)
    
    # Need to create a layout when doing separate calls to draw nodes and edges
    pos = nx.spring_layout(G,iterations=50)
    #pos = nx.random_layout(G)
    #pos = nx.kamada_kawai_layout(G)
    
    save_image(G,pos,fpath,colors,size)
    #plot_image(G,pos,fpath)
  
    
def save_image(G,pos,fpath,colors,size):
    
    import matplotlib.pyplot as plt
    import networkx as nx
    
    plt.figure(figsize=(30,25))
    
    nx.draw_networkx_nodes(G,pos,node_color=colors,node_size=max(100,min(2000,20000/size)))
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G,pos,font_size=max(20,min(50,1600/size)),font_color='k',font_family='sans-serif',font_weight='normal')
    
    plt.show(False)
    plt.savefig(fpath, dpi=300)


def plot_image(G,pos,fpath):       
    import plotly.graph_objects as go
    
    nodes = G.nodes(); edges = G.edges(); text = []
    
    edge_x = []; edge_y = []
    for edge in edges:
        node1,node2 = edge
        x0, y0 = pos[node1]
        x1, y1 = pos[node2]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        text.append(node1)
        text.append(node2)
        text.append(None)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []; node_y = []
    for node in nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='                   Minimum Spanning Tree<br>',
                titlefont_size=24,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    # Size points by the number of connections i.e. node_trace.marker.size = node_adjacencies
    node_adjacencies = []; node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    # Annotate your figure
    z = list(zip(edge_x,edge_y))
    for i, e in enumerate(z):
        if e[0] is not None:
            fig.add_annotation(x=e[0],y=e[1],text=text[i],font_size=10,
                               showarrow=False,xshift=-10)
    fig.show(False)
    fig.write_image(fpath)
    
def summary(components,b=False):
    import numpy as np
    all_sizes = [len(c) for c in sorted(components,key=len)]
    sizes = sorted(set(all_sizes))
    sizes = np.array(sizes)   
    for x in sizes:
        n = sum(all_sizes==x)
        if n == 1:
            print(f"{n} component of size {x}")
        else:
            print(f"{n} components of size {x}")
    if b: 
        print(components)
            
            
def getInfo(model):
    """
    Get information on components of a graph' model equations variables.
    
    Parameters:
        :param model: The Model object.
        :type model: Instance of class Model.
    """
    import networkx as nx
    
    m = dict()
    delimiters = "+","-","*","/","**","^", "(",")","="," "
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    eqs = model.symbolic.equations
    variable_names = model.symbols["variables"]
    eqsLabel = model.eqLabels
    print(f"\nNumber of equations {len(eqs)} and of endogenous variables {len(variable_names)}")
    for i,eq in enumerate(eqs):
        label = eqsLabel[i]
        if "=" in eq:
            ind   = eq.index("=")
            right = eq[1+ind:].strip()
        else:
            right = eq
        arr   = re.split(regexPattern,right)
        arr   = list(filter(None,arr))
        m[label] = set([x for x in arr if x in variable_names])
                
    nodes = dict((v,k) for k,v in enumerate(variable_names))
    
    from info.graph import Graph
    g = Graph(m=nodes)
    for k in m:
        for v in m[k]:
            g.addEdge(k,v)
    
    # # Depth-first search graph traversal
    # g.BCC() 
    # g.summary()
    
    
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for k in m:
        nds = m[k]
        for k2 in nds:
            G.add_edge(k,k2)
            
    connected_components = list(nx.connected_components(G.to_undirected()))
    print(f"\nNumber of connected components {len(connected_components)}")  
    summary(connected_components)
    
    strongly_connected_components = list(nx.strongly_connected_components(G))
    print(f"\nNumber of strongly connected components {len(strongly_connected_components)}")
    summary(strongly_connected_components)
    
    attracting_components = list(nx.attracting_components(G))
    print(f"\nNumber of attracting components {len(attracting_components)}")
    summary(attracting_components)
    
    biconnected_components = list(nx.biconnected_components(G.to_undirected()));
    print(f"\nNumber of biconnected components {len(biconnected_components)}")
    summary(biconnected_components)
    
    weakly_connected_components = list(nx.weakly_connected_components(G))
    print(f"\nNumber of weekly connected components {len(weakly_connected_components)}")
    summary(weakly_connected_components)

    

