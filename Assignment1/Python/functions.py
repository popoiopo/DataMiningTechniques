import pandas as pd
import numpy as np
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
import randomcolor
import math
from bokeh.models import HoverTool, LabelSet, SaveTool
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import numpy as np
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
import randomcolor
import math
from bokeh.models import HoverTool, LabelSet, SaveTool
from bokeh.plotting import figure


def printInfo(df, amount):
    print("LENGTH")
    print(len(df))
    print("*********************************************************")

    print("KEYS")
    print(len(df.keys()))
    print(df.keys())
    print("*********************************************************")

    print("UNIQUE VALUES COUNT")
    for key in df.keys():
        print(key)
        print(len([str(x).lower() for x in list(df[key].unique())]))
        print([str(x).lower() for x in list(df[key].unique())])
        print("")
    print("*********************************************************")

    # print("FIRST 5")
    # print(df.head(5))
    # print("*********************************************************")

    if amount == "all":
        print("Value_counts")
        for key in df.keys():
            print(df.groupby(key).count())
        print("*********************************************************")

        print("UNIQUE VALUES")
        for key in df.keys():
            print(key)
            print(list(df[key].unique()))
            print("")
        print("*********************************************************")

def getDataAttr(df):
    x = []
    y = []
    for key in df.keys():
        x.append(key)
        y.append(len([str(x).lower() for x in list(df[key].unique())]))
    return (x, y)

def vizBar(df):
    x, y = getDataAttr(df)
    labels = [str(i) for i in range(len(x))]

    output_file("bar_colors.html")

    rand_color = randomcolor.RandomColor()
    colors = rand_color.generate(count=16)
    source = ColumnDataSource(data=dict(y=y, x=x, labels=labels, color=colors))

    hover = HoverTool(tooltips=[
        ("Var", "@x"),
        ("Value", "@y"),
    ], mode='vline')

    p = figure(x_range=x, y_range=(0, max(y)+40), plot_height=400, title="Unique Answers/Question",
               tools=[hover, "save"])
    labels = LabelSet(x='x', y='y', text='labels', level='glyph',
                      x_offset=-6, y_offset=5, source=source, render_mode='canvas')

    p.vbar(x='x', top='y', width=0.9, color='color', source=source)

    p.xgrid.grid_line_color = None
    p.yaxis.major_label_orientation = "vertical"
    p.add_layout(labels)
    p.xaxis.visible = False
    show(p)


def vizPie(df, keys):
    counter = 0
    row = 2
    column = 3
    colors = ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf','#d9ef8b','#a6d96a','#66bd63','#1a9850']
    f, axarr = plt.subplots(2, 3)
    explode = (0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05)
    for r in range(row):
        for c in range(column):
            # Init data
            data = df[keys[counter]].value_counts()
            # Pie chart
            labels = data.keys()
            sizes = data.values
            explod = explode[:len(data.values)]
            axarr[r,c].pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90, explode = explod, pctdistance=0.85)
            axarr[r,c].set_title(keys[counter])
            axarr[r,c].axis('equal')
            counter += 1
    plt.tight_layout()
    plt.show()

























    # p = []
    # print(keys)

    # from math import pi

    # from bokeh.io import output_file, show
    # from bokeh.layouts import gridplot
    # from bokeh.models import ColumnDataSource
    # from bokeh.plotting import figure

    # output_file("pie.html")

    # for key in keys:
    #     source = ColumnDataSource(data=dict(
    #         start=[0, 0.2], end=[0.9, 2*pi], color=['firebrick', 'navy']
    #     ))

    #     plot = figure()
    #     plot.wedge(x=0, y=0, start_angle='start', end_angle='end', radius=0.4,
    #                color='color', alpha=0.6, source=source)
    #     plot.xgrid.grid_line_color = None

    #     p.append(plot)

    # grid = gridplot([[p[0], p[1], p[2]], [p[3], p[4], p[5]]])
    # show(grid)
