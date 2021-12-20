from flask import Flask, render_template, url_for ,request
import os 
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, PrintfTickFormatter
data=pd.read_csv('Height.csv')
app = Flask(__name__)

palette = ['#ba32a0', '#f85479', '#f8c260', '#00c2ba']

chart_font = 'Helvetica'
chart_title_font_size = '16pt'
chart_title_alignment = 'center'
axis_label_size = '14pt'
axis_ticks_size = '10pt'
default_padding = 30
chart_inner_left_padding = 0.015
chart_font_style_title = 'bold italic'


def palette_generator(length, palette):
    int_div = length // len(palette)
    remainder = length % len(palette)
    return (palette * int_div) + palette[:remainder]


def plot_styler(p):
    p.title.text_font_size = chart_title_font_size
    p.title.text_font  = chart_font
    p.title.align = chart_title_alignment
    p.title.text_font_style = chart_font_style_title
    p.y_range.start = 0
    p.x_range.range_padding = chart_inner_left_padding
    p.xaxis.axis_label_text_font = chart_font
    p.xaxis.major_label_text_font = chart_font
    p.xaxis.axis_label_standoff = default_padding
    p.xaxis.axis_label_text_font_size = axis_label_size
    p.xaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_text_font = chart_font
    p.yaxis.major_label_text_font = chart_font
    p.yaxis.axis_label_text_font_size = axis_label_size
    p.yaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_standoff = default_padding
    p.toolbar.logo = None
    p.toolbar_location = None

    # change appearance of legend text
    p.legend.label_text_font = "times"
    p.legend.label_text_font_style = "italic"
    p.legend.label_text_color = "navy"

    # change border and background of legend
    p.legend.border_line_width = 1
    p.legend.border_line_color = "navy"
    p.legend.border_line_alpha = 0.8
    p.legend.background_fill_color = "navy"
    p.legend.background_fill_alpha = 0.2
    p.legend.location = "bottom_right"

    p.xaxis.axis_label = 'index'
    p.yaxis.axis_label = 'Height (cm)'
    p.xgrid.visible = True
    p.ygrid.visible = True
    p.xgrid.grid_line_color="gray"
    p.ygrid.grid_line_color="gray"

color=['','red','blue','green','orange']
def height_plot(d):
    count=0
    hover_tool = HoverTool(
            tooltips=[('number', '@index'), ('height', '@height')])
    p = figure(tools=[hover_tool], plot_height=400,plot_width=1200, title='Height of soya plant')
    
    #plots=[]
    j=d
    if len(j)>=1:
        for i in j:
            count +=1
            print(1234)
            print(i)
            
            h=data[i].values
            
            ind=list(data[i].index)
            print(ind)
            source= ColumnDataSource({'height':h,'index':ind})
            
            
            p.line(x='index', y='height',source=source,color=color[count],legend_label=str(i))
            p.xaxis.ticker = source.data['index']
            
            p.xaxis.ticker = source.data['index']
            
            p.xaxis.ticker = source.data['index']
           
            
            plot_styler(p)
    return p

def redraw(df):
    height_chart = height_plot(df)
    print(height_chart)
    
    return height_chart





@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/analytics")   
def analytics():
    return render_template('analytics.html')

@app.route("/analytics", methods=['GET', 'POST'])
def plot():
    
    int_features = [x for x in request.form.keys()]
    if len(int_features)==0:
        
        return render_template('analytics.html')

    print(int_features)
    plot=redraw(int_features)

    plot_script, plot_div  = components(plot)
   
    kwargs = {'plot_script': plot_script, 'plot_div': plot_div}
    

    
    return render_template('chart.html', **kwargs)




if __name__ == '__main__':
    app.run(debug=True)

