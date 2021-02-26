import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import scipy.linalg as la
import pickle
import random
import numpy as np
import holoviews as hv
import networkx as nx
from bokeh.plotting import show
from holoviews import opts

hv.extension("bokeh")

class DI:
    def __init__(self, year):
        self.data = pickle.load(open('DIG_empirical_' + str(year) + '.pkl', 'rb' ))[0]
        self.firmlist = pickle.load(open('DIG_empirical_' + str(year) + '.pkl', 'rb' ))[1]
        self.threshold = np.mean(self.data)
        self.N = self.data.shape[0]
        self.year = year

    def binary(self, thresh):
        return (self.data > thresh) * 1

    def connectivity(self, thresh):
        return self.binary(thresh).sum().sum() / (self.binary(thresh).shape[0]) ** 2   #degree of granger causality

    def draw(self, thresh, nb_firms_to_draw, node_labels="Index"):
        """
        Draws the Directed Information Graph.
        :param DIG_data: Binary matrix. 1 = firm i influences firm j.
        :param node_labels: Node labels that will show on the graph. Can take as value : Index, TICKER, PERMCO, COMNAM
        :param nb_firms_to_draw: Number of nodes to show on the graph. It will show the first nb_firms_to_draw firms.
        :return: DIG
        """

        plt.figure()
        ax = plt.gca()
        ax.set_title('Network: ' + str(self.year - 4) + ' - ' +  str(self.year))
        G = nx.DiGraph()

        if node_labels == "Index":
            for i in range(nb_firms_to_draw):
                G.add_node(i)
                for j in range(nb_firms_to_draw):  # DIG.shape[1]
                    if self.binary(thresh)[i, j] == 1:
                        G.add_edge(i, j)
        else:
            for i in range(nb_firms_to_draw):
                G.add_node(self.firmlist[self.firmlist['big'] == True][node_labels].reset_index(drop=True)[i])
                for j in range(nb_firms_to_draw):  # DIG.shape[1]
                    if self.binary(thresh)[i, j] == 1:
                        G.add_edge(self.firmlist[self.firmlist['big'] == True][node_labels].reset_index(drop=True)[i],
                                   self.firmlist[self.firmlist['big'] == True][node_labels].reset_index(drop=True)[j])

        nx.draw(G, with_labels=True, font_weight='normal', ax=ax)
        _ = ax.axis('off')
        plt.show()


def DIG_labels(DIG,firmlist):
    DIG_binary_withlabels = pd.DataFrame(data = DIG, index = firmlist[firmlist['big']==True]['TICKER'])
    return DIG_binary_withlabels.T.set_index(firmlist[firmlist['big']==True]['TICKER']).T


if __name__ == "__main__":
    Networks = [1994,1999,2004,2009,2014,2019]
    DIG = []
    DIGbinary = []
    conn = []
    influencers = []
    influenced = []
    firmlists = []

    for year in Networks:
        time.sleep(5)
        DIGyear = DI(year)

        # interactive plot
        df = DIG_labels(DIGyear.data, DIGyear.firmlist)
        df.index.name = 'from'
        DI_vec = df.stack().reset_index().rename(columns={'TICKER': 'to', 0: 'DI'})
        DI_vec = DI_vec.drop(DI_vec[DI_vec['DI'] == 0].index)
        DI_vec['from'] = 'from: ' + DI_vec['from']
        DI_vec['to'] = 'to: ' + DI_vec['to']

        fromdf = pd.DataFrame({'from':['from: ' + e for e in df.index.to_list()], 'to':['from: ' + e for e in df.index.to_list()],'DI': 0})
        todf =  pd.DataFrame({'from':['to: ' + e for e in df.index.to_list()], 'to':['to: ' + e for e in df.index.to_list()],'DI': 0})
        DI_vec = DI_vec.append(fromdf)
        DI_vec = DI_vec.append(todf)

        firm_list = list(['from: ' + e for e in df.index.to_list()]) + list(['to: ' + e for e in df.index.to_list()])
        firm_dataset = hv.Dataset(pd.DataFrame(firm_list, columns=["Firms"]))

        firm_graph_with_labels = hv.Graph((DI_vec, firm_dataset), ["from", "to"])
        labels = hv.Labels(firm_graph_with_labels.nodes, ['x', 'y'], ['Firms'])
        labels.opts(xoffset=-0.00, yoffset=0.04, text_font_size='8pt', text_color="black")

        opts.defaults(opts.Graph(width=900, height=900, title='Network: ' + str(year - 4) + ' - ' +  str(year),
                                 xaxis=None, yaxis=None, show_frame=False, edge_alpha=0.6, node_color="dodgerblue",
                                 node_size=15, edge_line_width=1, edge_color="red",directed=False, bgcolor = "white"))
        graph = firm_graph_with_labels * labels
        show(hv.render(graph))
        #todo tooltip : https://stackoverflow.com/questions/59609911/holoviews-hovertool-show-extra-row
        #https://stackoverflow.com/questions/53680572/cannot-populate-hover-tooltip-values-in-bokeh-network-plot

        #hv.save(graph, 'Network'+str(year)+'.html', title = 'Network: ' + str(year - 4) + ' - ' +  str(year), embed =True)
        from bokeh.embed import file_html
        from bokeh.resources import CDN
        html = file_html(graph, "my plot")
        del DI_vec, df

        #append measures
        DIG.append(DIG_labels(DIGyear.data, DIGyear.firmlist)) #DI with comp names)
        DIGbinary.append(DIG_labels(DIGyear.binary(np.mean(DIGyear.data)), DIGyear.firmlist)) #binary DI with comp names
        #DIGyear.draw(DIGyear.threshold, DIGyear.N, "TICKER") #plot network
        conn.append(DIGyear.connectivity(DIGyear.threshold)) #connectiviy
        influencers.append(np.sum(DIG_labels(DIGyear.binary(DIGyear.threshold),DIGyear.firmlist),axis=1).sort_values(ascending = False)) #topinfluencers
        influenced.append(np.sum(DIG_labels(DIGyear.binary(DIGyear.threshold),DIGyear.firmlist),axis=0).sort_values(ascending = False)) #topinfluenced





