#Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import os
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from my_model import detect_image_
from datetime import datetime
import tensorflow as tf
from object_detection.utils import label_map_util

#load model
#tf.keras.backend.clear_session()

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

model =load_model("model/leaf.h5")
pod_model = tf.saved_model.load('model/pod_model/')
flower_model = tf.saved_model.load('model/flower_model/')
print('@@ Model loaded')


def pred_cot_dieas(cott_plant):
  test_image = load_img(cott_plant, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)


  pred = np.argmax(result) # get the index of max value
  print('@@ pred = ', pred)
  if pred == 0:
      return "Unhealthy",'disease_plant.html' 
  else:
      return "Healthy", 'healthy_plant.html'

#------------>>pred_cot_dieas<<--end
# Create flask instance
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
            
            h=data[i]
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

# render index.html page
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

# render index.html page
@app.route("/index", methods=['GET', 'POST'])
def index():
        return render_template('index.html')


# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])

def predict():
  
  if request.method == 'POST':
    file = request.files['image'] # fet input
    model_name=request.form.get("model_name")
    filename = file.filename        
       
    file_path = os.path.join('static/user uploaded', filename)
    file.save(file_path)
    input_img=cv2.imread(file_path)
    #input_img=cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    print("@@ Predicting class......")
    print(model_name)
    l=model_name.split(" ")
    print(l)
    r=[]
    for model_name in l:
      if model_name=="leaf":
        value, output_page = pred_cot_dieas(cott_plant=file_path)
        output_path=file_path
        time=datetime.now()
        with open("static/result.csv","a") as f:
          f.write("\n"+str(time)+","+str(value))
          r.append(value)
      elif model_name=="pod":
        output_page="pod.html"
        labelmap_path='static/pod/label_map.pbtxt'
        category_index =label_map_util.create_category_index_from_labelmap(labelmap_path)
        I,b,score=detect_image_(file_path,pod_model,category_index)
        print(score)
        print(type(score))
        if len(score)>0 and score[0]>0.90:
          value=str(len(b)*3)
          r.append(value)
          time=datetime.now()
          output_img=cv2.cvtColor(I,cv2.COLOR_RGB2BGR)
          output_path="static/pod/output.jpg"
          cv2.imwrite(output_path,output_img)
          with open("static/result.csv","a") as f:
            f.write(","+str(value))
        else:
          value=str(0)
          r.append(value)
          time=datetime.now()
          #output_img=cv2.cvtColor(I,cv2.COLOR_RGB2BGR)
          output_path="static/flower/output.jpg"
          cv2.imwrite(output_path,input_img)
          with open("static/result.csv","a") as f:
            f.write(","+str(value))

      elif model_name=="flower":
        output_page="pod.html"
        labelmap_path='static/flower/label_map.pbtxt'
        category_index =label_map_util.create_category_index_from_labelmap(labelmap_path) 
        I,b,score=detect_image_(file_path,flower_model,category_index)
        print(score)
        print(type(score))
        if len(score)>0 and score[0]>0.99:
          value=str(len(b))
          r.append(value)
          time=datetime.now()
          output_img=cv2.cvtColor(I,cv2.COLOR_RGB2BGR)
          output_path="static/flower/output.jpg"
          cv2.imwrite(output_path,output_img)
          with open("static/result.csv","a") as f:
            f.write(","+str(value))
        else:
          value=str(0)
          r.append(value)
          time=datetime.now()
          output_img=cv2.cvtColor(I,cv2.COLOR_RGB2BGR)
          output_path="static/flower/output.jpg"
          cv2.imwrite(output_path,input_img)
          with open("static/result.csv","a") as f:
            f.write(","+str(value))                    
      else:
        pass
      str(r)
    return render_template(output_page, pred_output = r, user_image = output_path)

# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,debug=True) 
    
    
