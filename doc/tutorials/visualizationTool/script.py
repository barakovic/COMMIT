#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import logging
import os


import nibabel as nib
import numpy as np
import pickle
import dipy
from dipy.viz import window, actor, ui
from dipy.tracking.streamline import transform_streamlines
import copy
import os
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import sys

DESCRIPTION = """
    Description....
        An interactive visualization tool to study convergence of COMMIT's microstructure estimates

    References
    ----------
    .. [1] ...

    .. [2] ...


    """

def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('commitOutputPath', action='store',
    		help='Path of the form "/home/user/.../CommitOutput"')

    p.add_argument('streamlinesNumber', action='store',
		help='the number of streamlines you want to compute')
	
    return p



parser = buildArgsParser()
args = parser.parse_args()
model = None
bar = None
lut_cmap = None
list_x_file = None
max_weight = None
saturation = None
renderer = None
norm_fib = None
norm1 = None
norm2 = None
norm3 = None
big_stream_actor = None
good_stream_actor = None
weak_stream_actor = None
big_Weight = None
good_Weight = None
weak_Weight = None
smallBundle_safe = None
smallWeight_safe = None
show_m = None
big_bundle = None
good_bundle = None
weak_bundle = None
nF = None
nIC = None
Ra = None
change_colormap_slider = None
remove_small_weights_slider = None
opacity_slider = None
remove_big_weights_slider = None
change_iteration_slider = None
num_computed_streamlines = None
numbers_of_streamlines_in_interval = None

def main():
    global parser
    global args
    global model
    global bar
    global lut_cmap
    global list_x_file
    global max_weight
    global saturation
    global renderer
    global norm_fib
    global norm1
    global norm2
    global norm3
    global big_stream_actor
    global good_stream_actor
    global weak_stream_actor
    global big_Weight
    global good_Weight
    global weak_Weight
    global smallBundle_safe
    global smallWeight_safe
    global show_m
    global big_bundle
    global good_bundle
    global weak_bundle
    global nF
    global nIC
    global Ra
    global change_colormap_slider
    global remove_small_weights_slider
    global opacity_slider
    global remove_big_weights_slider
    global change_iteration_slider
    global num_computed_streamlines
    global numbers_of_streamlines_in_interval
    
    #defining the model used (Stick or cylinder)
    model = None
    if(os.path.isdir(args.commitOutputPath+"/Results_StickZeppelinBall") and os.path.isdir(args.commitOutputPath+"/Results_CylinderZeppelinBall")):
        model_index = input("Which model do you want to load (1 for 'Cylinder', 2 for 'Stick') : ")
        if(model_index==1): model = "Cylinder"
        else: model ="Stick"
    elif(os.path.isdir(args.commitOutputPath+"/Results_StickZeppelinBall")):
        model = "Stick"
    elif(os.path.isdir(args.commitOutputPath+"/Results_CylinderZeppelinBall")):
        model = "Cylinder"
    else:
        print("No valide model in this path")
        sys.exit(0)


    #formalizing the filenames of the iterations
    list_x_file = [file for file in os.listdir(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/") if (file.endswith('.npy') and (file[:-4]).isdigit() )]
    normalize_file_name(list_x_file)
    list_x_file.sort()
    num_iteration=len(list_x_file)

    #number of streamlines we want to load
    num_computed_streamlines = int(args.streamlinesNumber)
    #computing interval of weights
    max_weight = 0;
    if(model == "Cylinder"):
        file = open( args.commitOutputPath+"/Results_"+model+"ZeppelinBall/results.pickle",'rb' )
        object_file = pickle.load( file )

        Ra = np.linspace( 0.75,3.5,12 ) * 1E-6

        nIC = len(Ra)    # IC  atoms
        nEC = 4          # EC  atoms
        nISO = 1         # ISO atoms

        nF = object_file[0]['optimization']['regularisation']['sizeIC']
        nE = object_file[0]['optimization']['regularisation']['sizeEC']
        nV = object_file[0]['optimization']['regularisation']['sizeISO']


        num_ADI = np.zeros( nF )
        den_ADI = np.zeros( nF )

        dim = nib.load(args.commitOutputPath+"/Results_"+model+"ZeppelinBall/compartment_IC.nii.gz").get_data().shape
        norm_fib = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm_fib.npy")
        norm1 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm1.npy")
        norm2 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm2.npy")
        norm3 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm3.npy")
        for itNbr in list_x_file:
            #computing diameter
            x = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/"+ itNbr +'.npy')
            x_norm = x / np.hstack( (norm1*norm_fib,norm2,norm3) )

            for i in range(nIC):
                den_ADI = den_ADI + x_norm[i*nF:(i+1)*nF]
                num_ADI = num_ADI + x_norm[i*nF:(i+1)*nF] * Ra[i]

            Weight = 2 * ( num_ADI / ( den_ADI + np.spacing(1) ) ) * 1E6
            smallWeight_safe = Weight[:num_computed_streamlines]
            itNbr_max = np.amax(smallWeight_safe)
            if(itNbr_max>max_weight):
                max_weight=itNbr_max
    else:#model==Stick
        file = open( args.commitOutputPath+"/Results_"+model+"ZeppelinBall/results.pickle",'rb' )
        object_file = pickle.load( file )
        norm_fib = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm_fib.npy")
        norm1 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm1.npy")
        norm2 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm2.npy")
        norm3 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm3.npy")
        nF = object_file[0]['optimization']['regularisation']['sizeIC']
        for itNbr in list_x_file:
            x = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/"+ itNbr +'.npy')
            x_norm = x / np.hstack( (norm1*norm_fib,norm2,norm3) )

            Weight = x_norm[:nF]  #signal fractions
            smallWeight_safe = Weight[:num_computed_streamlines]
            itNbr_max = np.amax(smallWeight_safe)
            if(itNbr_max>max_weight):
                max_weight=itNbr_max
    #we need an interval slightly bigger than the max_weight
    max_weight = max_weight + 0.00001

    #computing initial weights
    if(model == "Cylinder"):#model==Cylinder
        file = open( args.commitOutputPath+"/Results_"+model+"ZeppelinBall/results.pickle",'rb' )
        object_file = pickle.load( file )

        Ra = np.linspace( 0.75,3.5,12 ) * 1E-6

        nIC = len(Ra)    # IC  atoms
        nEC = 4          # EC  atoms
        nISO = 1         # ISO atoms

        nF = object_file[0]['optimization']['regularisation']['sizeIC']
        nE = object_file[0]['optimization']['regularisation']['sizeEC']
        nV = object_file[0]['optimization']['regularisation']['sizeISO']

        dim = nib.load(args.commitOutputPath+"/Results_"+model+"ZeppelinBall/compartment_IC.nii.gz").get_data().shape


        norm_fib = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm_fib.npy")
        #add the normalisation
        x = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/"+list_x_file[0]+'.npy')
        norm1 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm1.npy")
        norm2 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm2.npy")
        norm3 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm3.npy")
        x_norm = x / np.hstack( (norm1*norm_fib,norm2,norm3) )

        num_ADI = np.zeros( nF )
        den_ADI = np.zeros( nF )

        for i in range(nIC):
            den_ADI = den_ADI + x_norm[i*nF:(i+1)*nF]
            num_ADI = num_ADI + x_norm[i*nF:(i+1)*nF] * Ra[i]

        Weight = 2 * ( num_ADI / ( den_ADI + np.spacing(1) ) ) * 1E6
        smallWeight_safe = Weight[:num_computed_streamlines]
        weak_Weight = smallWeight_safe[:1]
        big_Weight = smallWeight_safe[:1]
        good_Weight = copy.copy(smallWeight_safe)
    else:#model==Stick
        file = open( args.commitOutputPath+"/Results_"+model+"ZeppelinBall/results.pickle",'rb' )
        object_file = pickle.load( file )
        nF = object_file[0]['optimization']['regularisation']['sizeIC']
        x = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/"+list_x_file[0]+'.npy')
        norm1 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm1.npy")
        norm2 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm2.npy")
        norm3 = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/norm3.npy")
        x_norm = x / np.hstack( (norm1*norm_fib,norm2,norm3) )

        Weight = x_norm[:nF]  #signal fractions
        smallWeight_safe = Weight[:num_computed_streamlines]
        weak_Weight = smallWeight_safe[:1]
        big_Weight = smallWeight_safe[:1]
        good_Weight = copy.copy(smallWeight_safe)

    #load streamlines from the dictionary_TRK_fibers_trk file
    streams, hdr = nib.trackvis.read(args.commitOutputPath+"/dictionary_TRK_fibers.trk")
    streamlines = [s[0] for s in streams]
    smallBundle_safe = streamlines[:num_computed_streamlines]
    weak_bundle = smallBundle_safe[:1]
    big_bundle = smallBundle_safe[:1]
    good_bundle = copy.copy(smallBundle_safe)
    #number of good streamlines
    num_streamlines = len(smallBundle_safe)


    # mapping streamlines and initial weights(with a red bar) in a renderer
    hue = [0, 0]  # red only
    saturation = [0.0, 1.0]  # black to white

    lut_cmap = actor.colormap_lookup_table(
        scale_range=(0, max_weight),
        hue_range=hue,
        saturation_range=saturation)

    weak_stream_actor = actor.line(weak_bundle, weak_Weight,
                                   lookup_colormap=lut_cmap)
    big_stream_actor = actor.line(big_bundle, big_Weight,
                                lookup_colormap=lut_cmap)
    good_stream_actor = actor.line(good_bundle, good_Weight,
                               lookup_colormap=lut_cmap)

    bar = actor.scalar_bar(lut_cmap, title = 'weight')
    bar.SetHeight(0.5)
    bar.SetWidth(0.1)
    bar.SetPosition(0.85,0.45)

    renderer = window.Renderer()

    renderer.set_camera(position=(-176.42, 118.52, 128.20),
                        focal_point=(113.30, 100, 76.56),
                        view_up=(0.18, 0.00, 0.98))

    renderer.add(big_stream_actor)
    renderer.add(good_stream_actor)
    renderer.add(weak_stream_actor)
    renderer.add(bar)

    #adding sliders and renderer to a ShowManager
    show_m = window.ShowManager(renderer, size=(1200, 900))
    show_m.initialize()

    save_one_image_bouton = ui.LineSlider2D(min_value=0,
                                    max_value=1,
                                    initial_value=0,
                                    text_template="save",
                                    length=1)

    add_graph_bouton = ui.LineSlider2D(min_value=0,
                                    max_value=1,
                                    initial_value=0,
                                    text_template="graphe",
                                    length=1)

    color_slider = ui.LineSlider2D(min_value=0.0,
                                     max_value=1.0,
                                     initial_value=0,
                                     text_template="{value:.1f}",
                                     length=140)

    change_colormap_slider = ui.LineSlider2D(min_value=0,
                                    max_value=1.0,
                                    initial_value=0,
                                    text_template="",
                                    length=40)

    change_iteration_slider = ui.LineSlider2D(min_value=0,
                    #we can't have max_value=num_iteration because
                    #list_x_file[num_iteration] lead to an error
                                    max_value=num_iteration-0.01,
                                    initial_value=0,
                                    text_template=list_x_file[0],
                                    length=140)

    remove_big_weights_slider = ui.LineSlider2D(min_value=0,
                                    max_value=max_weight,
                                    initial_value=max_weight,
                                    text_template="{value:.2f}",
                                    length=140)

    remove_small_weights_slider = ui.LineSlider2D(min_value=0,
                                    max_value=max_weight,
                                    initial_value=0,
                                    text_template="{value:.2f}",
                                    length=140)

    opacity_slider = ui.LineSlider2D(min_value=0.0,
                                     max_value=1.0,
                                     initial_value=0.5,
                                     text_template="{ratio:.0%}",
                                     length=140)

    save_one_image_bouton.add_callback(save_one_image_bouton.slider_disk,
                                "LeftButtonPressEvent", save_one_image)

    color_slider.add_callback(color_slider.slider_disk,
                                "MouseMoveEvent", change_streamlines_color)
    color_slider.add_callback(color_slider.slider_line,
                               "LeftButtonPressEvent", change_streamlines_color)
    add_graph_bouton.add_callback(add_graph_bouton.slider_disk,
                                "LeftButtonPressEvent", add_graphe)

    change_colormap_slider.add_callback(change_colormap_slider.slider_disk,
                                "MouseMoveEvent", change_colormap)
    change_colormap_slider.add_callback(change_colormap_slider.slider_line,
                                "LeftButtonPressEvent", change_colormap)
    change_iteration_slider.add_callback(change_iteration_slider.slider_disk,
                                "MouseMoveEvent", change_iteration)
    change_iteration_slider.add_callback(change_iteration_slider.slider_line,
                               "LeftButtonPressEvent", change_iteration)

    remove_big_weights_slider.add_callback(remove_big_weights_slider.slider_disk,
                                "MouseMoveEvent", remove_big_weight)
    remove_big_weights_slider.add_callback(remove_big_weights_slider.slider_line,
                               "LeftButtonPressEvent", remove_big_weight)

    remove_small_weights_slider.add_callback(remove_small_weights_slider.slider_disk,
                                "MouseMoveEvent", remove_small_weight)
    remove_small_weights_slider.add_callback(remove_small_weights_slider.slider_line,
                               "LeftButtonPressEvent", remove_small_weight)
    opacity_slider.add_callback(opacity_slider.slider_disk,
                                "MouseMoveEvent", change_opacity)
    opacity_slider.add_callback(opacity_slider.slider_line,
                               "LeftButtonPressEvent", change_opacity)

    color_slider_label = ui.TextBlock2D()
    color_slider_label.message = 'color of streamlines'

    change_colormap_slider_label_weight = ui.TextBlock2D()
    change_colormap_slider_label_weight.message = 'weight color'
    change_colormap_slider_label_direction = ui.TextBlock2D()
    change_colormap_slider_label_direction.message = 'direction color'

    change_iteration_slider_label = ui.TextBlock2D()
    change_iteration_slider_label.message = 'number of the iteration'

    remove_big_weights_slider_label = ui.TextBlock2D()
    remove_big_weights_slider_label.message = 'big weights subdued'

    remove_small_weights_slider_label = ui.TextBlock2D()
    remove_small_weights_slider_label.message = 'small weights subdued'

    opacity_slider_label = ui.TextBlock2D()
    opacity_slider_label.message = 'Unwanted weights opacity'

    numbers_of_streamlines_in_interval = ui.TextBlock2D()
    numbers_of_streamlines_in_interval.message = "Number of streamlines in interval: "+str(num_streamlines)


    panel = ui.Panel2D(center=(300, 160),
                       size=(500, 280),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")

    panel.add_element(save_one_image_bouton, 'relative', (0.9, 0.9))
    panel.add_element(add_graph_bouton, 'relative', (0.9, 0.77))
    panel.add_element(color_slider_label, 'relative', (0.05, 0.85))
    panel.add_element(color_slider, 'relative', (0.7, 0.9))
    panel.add_element(numbers_of_streamlines_in_interval, 'relative', (0.05, 0.72))
    panel.add_element(change_colormap_slider_label_weight, 'relative', (0.05, 0.59))
    panel.add_element(change_colormap_slider_label_direction, 'relative', (0.5, 0.59))
    panel.add_element(change_colormap_slider, 'relative', (0.4, 0.64))
    panel.add_element(change_iteration_slider_label, 'relative', (0.05, 0.46))
    panel.add_element(change_iteration_slider, 'relative', (0.7, 0.51))
    panel.add_element(remove_big_weights_slider_label, 'relative', (0.05, 0.33))
    panel.add_element(remove_big_weights_slider, 'relative', (0.7, 0.37))
    panel.add_element(remove_small_weights_slider_label, 'relative', (0.05, 0.2))
    panel.add_element(remove_small_weights_slider, 'relative', (0.7, 0.24))
    panel.add_element(opacity_slider_label, 'relative', (0.05, 0.07))
    panel.add_element(opacity_slider, 'relative', (0.7, 0.11))

    panel.add_to_renderer(renderer)
    renderer.reset_clipping_range()

    show_m.render()
    show_m.start()



#some usefull functions:

def normalize_file_name(list):
    """the iterations of COMMIT's files names has to be of the form xxxx.npy

    Parameters
    ----------
    list : list
        a list of each iteration's file that we will normalize
    """
    i=0
    while(i<len(list)):
        if list[i].endswith('.npy'):
            list[i] = list[i][:-4]
            os.rename(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/"+list[i]+'.npy', args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/"+list[i].zfill(4)+'.npy')
        i=i+1
    return
    
#to refresh the ShowManager each time there is a visual change
def refresh_showManager(i_ren, obj, slider):

    global show_m
    show_m.render()
    return

#function called by the graphe's slider
def add_graphe(i_ren, obj, slider):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, rectangles = ax.hist(x=smallWeight_safe, bins=50, density=False, label = 'graphe')
    plt.xlabel('weight')
    plt.ylabel('number of streamlines')
    plt.title('histogram of iteration '+str(list_x_file[int(change_iteration_slider.value)]))
    plt.grid()
    plt.show()
    refresh_showManager(i_ren, obj, slider)
    return


#function to separate the streamlines in 3x2 arrayes according to the sliders
def refresh_3x2_arrays():
    global weak_Weight
    global weak_bundle
    global big_Weight
    global big_bundle
    global good_Weight
    global good_bundle

    big_bundle = []
    big_Weight = []
    good_bundle = []
    good_Weight = []
    weak_bundle = []
    weak_Weight = []

    i=0
    while(i<len(smallWeight_safe)):
        if(smallWeight_safe[i]<remove_small_weights_slider.value):
            weak_Weight.append(smallWeight_safe[i])
            weak_bundle.append(smallBundle_safe[i])
        elif(smallWeight_safe[i]>remove_big_weights_slider.value):
            big_Weight.append(smallWeight_safe[i])
            big_bundle.append(smallBundle_safe[i])
        else:
            good_Weight.append(smallWeight_safe[i])
            good_bundle.append(smallBundle_safe[i])
        i=i+1
    return

#function to refresh the 3 (weak, good, big) actors in the renderer using the 3x2 arrays
def refresh_3_stream_actors():
    global big_stream_actor
    global good_stream_actor
    global weak_stream_actor
    
    #removing the out of date actors
    renderer.RemoveActor(big_stream_actor)
    renderer.RemoveActor(good_stream_actor)
    renderer.RemoveActor(weak_stream_actor)
    
    #adding the 3 (weak, good, big) actors to the renderer 
    if(change_colormap_slider.value<=change_colormap_slider.max_value/2):
        if(len(good_Weight)>0):
            good_stream_actor = actor.line(good_bundle, good_Weight,
                              lookup_colormap=lut_cmap)
            renderer.add(good_stream_actor)
        if(len(weak_Weight)>0):
            weak_stream_actor = actor.line(weak_bundle, weak_Weight,
                              opacity =opacity_slider.value,
                              lookup_colormap=lut_cmap)
            renderer.add(weak_stream_actor)
        if(len(big_Weight)>0):
            big_stream_actor = actor.line(big_bundle, big_Weight,
                              opacity =opacity_slider.value,
                              lookup_colormap=lut_cmap)
            renderer.add(big_stream_actor)
    else:
        if(len(good_Weight)>0):
            good_stream_actor = actor.line(good_bundle)
            renderer.add(good_stream_actor)
        if(len(weak_Weight)>0):
            weak_stream_actor = actor.line(weak_bundle,
                              opacity =opacity_slider.value)
            renderer.add(weak_stream_actor)
        if(len(big_Weight)>0):
            big_stream_actor = actor.line(big_bundle,
                              opacity =opacity_slider.value)
            renderer.add(big_stream_actor)
    return
    
#function called by the colormap's slider
def change_colormap(i_ren, obj, slider):

    if(slider.value<=slider.max_value/2):
        bar.SetVisibility(True)
    else:
        bar.SetVisibility(False)
    
    refresh_3_stream_actors()
    refresh_showManager(i_ren, obj, slider)
    return

        
#function called by the iteration's slider
def change_iteration(i_ren, obj, slider):
    global Weight
    global smallWeight_safe
    
    if(model == "Cylinder"):
        #load the weights of the correct iteration according to the slider
        x = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/"+list_x_file[int(slider.value)]+'.npy')
        x_norm = x / np.hstack( (norm1*norm_fib,norm2,norm3) )

        num_ADI = np.zeros( nF )
        den_ADI = np.zeros( nF )

        for i in range(nIC):
            den_ADI = den_ADI + x_norm[i*nF:(i+1)*nF]
            num_ADI = num_ADI + x_norm[i*nF:(i+1)*nF] * Ra[i]
    
        Weight = 2 * ( num_ADI / ( den_ADI + np.spacing(1) ) ) * 1E6
        smallWeight_safe = Weight[:num_computed_streamlines]
    else:#model==Stick
        #load the weights of the correct iteration according to the slider
        x = np.load(args.commitOutputPath+"/Coeff_x_"+model+"ZeppelinBall/"+list_x_file[int(slider.value)]+'.npy')
        x_norm = x / np.hstack( (norm1*norm_fib,norm2,norm3) )

        Weight = x_norm[:nF]  #signal fractions
        smallWeight_safe = Weight[:num_computed_streamlines]
    
    refresh_3x2_arrays()
    refresh_3_stream_actors()
    #updating the number of good streamlines and the name of the iteration slider
    numbers_of_streamlines_in_interval.message = "Number of streamlines in interval: "+str(len(good_Weight))
    slider.text_template=list_x_file[int(slider.value)]
    refresh_showManager(i_ren, obj, slider)
    return

    
#function called by the remove_big_weight's slider
def remove_big_weight(i_ren, obj, slider):
    
    refresh_3x2_arrays()
    refresh_3_stream_actors()
    #updating the number of good streamlines
    numbers_of_streamlines_in_interval.message = "Number of streamlines in interval: "+str(len(good_Weight)) 
    refresh_showManager(i_ren, obj, slider)
    return

    
#function called by the remove_small_weight's slider
def remove_small_weight(i_ren, obj, slider):
    
    refresh_3x2_arrays() 
    refresh_3_stream_actors()
    #updating the number of good streamlines
    numbers_of_streamlines_in_interval.message = "Number of streamlines in interval: "+str(len(good_Weight))
    refresh_showManager(i_ren, obj, slider)
    return


#function called by the change_opacity's slider
def change_opacity(i_ren, obj, slider):
    
    refresh_3_stream_actors()
    refresh_showManager(i_ren, obj, slider)
    return

            
#change the color of the streamlines
def change_streamlines_color(i_ren, obj, slider):
    global hue
    global lut_cmap

    #refreshing the hue and lut_cmap
    hue = [0, slider.value]
    lut_cmap = actor.colormap_lookup_table(
        scale_range=(0, max_weight),
        hue_range=hue,
        saturation_range=saturation)
    
    #refreshing the bar
    bar.SetLookupTable(lut_cmap)
    
    refresh_3_stream_actors()
    refresh_showManager(i_ren, obj, slider)
    return
            
    
#save the current view
def save_one_image(i_ren, obj, slider):
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(show_m.window)
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()
    h, w, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    arr = vtk_to_numpy(vtk_array).reshape(h, w, components)

    writer = vtk.vtkPNGWriter()
    fileName = "saved_iteration_"+list_x_file[int(change_iteration_slider.value)]+".png"
    i = 0
    while(os.path.isfile(fileName)):
        i=i+1
        fileName = "saved_iteration_"+list_x_file[int(change_iteration_slider.value)]+"("+str(i)+").png"
    writer.SetFileName(fileName)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()
    refresh_showManager(i_ren, obj, slider)
    return

if __name__ == "__main__":
	main()
