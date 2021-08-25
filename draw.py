# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:16:11 2019

@author: tomson
"""

import numpy as np
import matplotlib
#matplotlib.use('PS')
#mpl.use("Qt5Agg")
#matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
# 在我的 notebook 里，要设置下面两行才能显示中文
#plt.rcParams['font.family'] = ['sans-serif']
## 如果是在 PyCharm 里，只要下面一行，上面的一行可以删除
#plt.rcParams['font.sans-serif'] = ['SimHei']

def heatmap(data, row_labels, col_labels,norm, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
#    im = ax.pcolor(data, edgecolors='k', linewidths=4)
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar=None
#    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#    cbar = ax.figure.colorbar(im, ax=ax,vmin=0,vmax=1, **cbar_kw)
#    cbar.
#    cbar.set_ticks(np.linspace(0, 1,5))
#    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels,fontsize=23)
    ax.set_yticklabels(row_labels,fontsize=23)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
#vegetables = ["cucumber", "tomato"]
#farmers = ["Farmer Joe", "Upland Bros."]
#
#harvest = np.array([[0,0.7],[0.7,0]])
#fig= plt.figure()
#ax=fig.add_subplot(111)
#
#im, cbar = heatmap(harvest, vegetables, farmers, None,ax=ax,
#                   cmap="YlGn", cbarlabel="harvest [t/year]")
#ax.set_yticklabels(farmers,fontsize=18)
##texts = annotate_heatmap(im, valfmt="{x:.1f} t")
#
#fig.tight_layout()
##plt.show()
#fig.savefig('./pics/pic'+'hjhj'+'.pdf',dpi=100)
#plt.show()
def draw_attention(words1,words2,att,name='hj'):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    assert len(words1)==att.shape[0]
    assert len(words2)==att.shape[1]
    fig, ax = plt.subplots(figsize = (20, 20))
    
    im, cbar = heatmap(att, words1, words2,norm, ax=ax,
                       cmap="YlGn", cbarlabel="Attention Value")
    #texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    
    fig.tight_layout()
    fig.savefig('./pics/pic'+name+'.eps',dpi=50)
#    plt.show()
#draw_attention(vegetables,farmers,harvest)