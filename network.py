import scrapper 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm

def create_edges(data, u1,u2):
    edges = []
    overall_weight = 0
    for collab in data.collabs[u1][u2].items():
        #print(collab)
        attr_dict = {'subject':collab[0], 'weight':collab[1]}
        edge = (u1, u2, attr_dict)
        edges.append(edge)

        overall_weight+=collab[1]
    #Edge with topic 'All'

    #Edge with topic 'All'
    overall_edge = (u1,u2, {'subject':'All','weight':overall_weight})
    edges.append(overall_edge)
    return edges

def subjects_distribution(subjects, first_n):
    distrib = {}
    for subject, weight in subjects:
        if subject != 'All':
            distrib[subject] = distrib.get(subject,0) + weight
    width = 0.8 # width of the bars

    fig, ax = plt.subplots()

    ordered_items = sorted(distrib.items(),key=lambda x: x[1], reverse = True)[:first_n]
    xvalues = [subject for subject, value in ordered_items]
    yvalues = [value for subject, value in ordered_items]

    pprint(xvalues)



    rects1 = ax.bar(xvalues, yvalues, width, color='b')
    ax.set_title("Subject histogram")
    ax.set_xlabel("Subject")
    ax.set_xticks(range(len(xvalues)),xvalues, rotation = 90) # set the position of the x ticks
    ax.set_ylabel("# of link with this topic")
    plt.show()
        


def degree_distribution(G):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()
            


data = scrapper.Scraper()
data.load_universities("universities.json")
data.load_collabs("collabs.json")
#scraper.print_stats()

inserted_universities = {}
edges = []
nodes = []


for u1 in data.collabs:
    inserted_universities[u1] = True
    nodes.append(u1)
    for u2 in data.collabs[u1]:
        if not inserted_universities.get(u2,False):
            tmp_edges = create_edges(data,u1,u2)
            edges.extend(tmp_edges)

G = nx.MultiGraph()
G.add_edges_from(edges)

###THERE ARE 110 nodes that don't connect with anything soo
not_connected_nodes = set(nodes)-set(G.nodes())
########################################################


#Analyzing subjects
subjects = [(attr['subject'], attr['weight']) for node1, node2, attr in G.edges(data=True)]
first_n = 10
subjects_distribution(subjects, first_n)
subject_set = set([subj for subj, weight in subjects])



degree_distribution(G)

        