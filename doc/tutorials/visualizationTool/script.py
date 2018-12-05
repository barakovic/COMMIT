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


DESCRIPTION = """
    Description....
        Visualize the diameters of streamlines.
        diameter computed with a .npy file and streamlines computed with a .trk file

    References
    ----------
    .. [1] ...

    .. [2] ...


    """

def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('dictionary_TRK_fibers_trk', action='store',
    		help='Path of the form "/home/user/.../dictionary_TRK_fibers.trk"')

    p.add_argument('results_pickleFilePath', action='store',
    		help='Path of the form "/home/user/.../results.pickle"')

    p.add_argument('compartment_IC_nii_gzFilePath', action='store',
    		help='Path of the form "/home/user/.../compartment_IC.nii.gz"')

    p.add_argument('norm_fib_npyFilePath', action='store',
    		help='Path of the form "/home/user/.../norm_fib.npy"')

    p.add_argument('weightsPath', action='store',
    		help='Path of the form "/home/user/.../iterations" with only the .npy files storing the diameters, named with the number of the iteration (from 1.npy to 9999.npy)')

    p.add_argument('streamlinesNumber', action='store',
		help='the number of streamlines you want to compute')
	
    return p

parser = buildArgsParser()
args = parser.parse_args()


def main():
    global parser
    global args


#the weight files names has to be of the form xxxx.npy
def normalize_file_name(list):
    i=0
    while(i<len(list)):
        if list[i].endswith('.npy'):
            list[i] = list[i][:-4]
            os.rename(args.weightsPath+'/'+list[i]+'.npy', args.weightsPath+'/'+list[i].zfill(4)+'.npy')
        i=i+1
    return

### formalizing the files names, load the streamlines and initial weights

list_x_file = [file for file in os.listdir(args.weightsPath+'/') if file.endswith('.npy')]
normalize_file_name(list_x_file)
list_x_file.sort()
num_iteration=len(list_x_file)
#number of streamlines we want to load
num_computed_streamlines = int(args.streamlinesNumber)


#computing diameter
file = open( args.results_pickleFilePath,'rb' )
object_file = pickle.load( file )

Ra = np.linspace( 0.75,3.5,12 ) * 1E-6

nIC = len(Ra)    # IC  atoms
nEC = 4          # EC  atoms
nISO = 1         # ISO atoms

nF = object_file[0]['optimization']['regularisation']['sizeIC']
nE = object_file[0]['optimization']['regularisation']['sizeEC']
nV = object_file[0]['optimization']['regularisation']['sizeISO']

dim = nib.load(args.compartment_IC_nii_gzFilePath).get_data().shape

norm_fib = np.load(args.norm_fib_npyFilePath)

x_norm = np.load(args.weightsPath+'/'+list_x_file[0]+'.npy')

num_ADI = np.zeros( nF )
den_ADI = np.zeros( nF )

for i in range(nIC):
    den_ADI = den_ADI + x_norm[i*nF:(i+1)*nF]
    num_ADI = num_ADI + x_norm[i*nF:(i+1)*nF] * Ra[i]
    
ADI = 2 * ( num_ADI / ( den_ADI + np.spacing(1) ) ) * 1E6
smallADI_safe = ADI[:num_computed_streamlines]
weak_ADI = smallADI_safe[:1]
big_ADI = smallADI_safe[:1]
good_ADI = copy.copy(smallADI_safe)

#computing streamlines
streams, hdr = nib.trackvis.read(args.dictionary_TRK_fibers_trk)
streamlines = [s[0] for s in streams]
smallBundle_safe = streamlines[:num_computed_streamlines]
weak_bundle = smallBundle_safe[:1]
big_bundle = smallBundle_safe[:1]
good_bundle = copy.copy(smallBundle_safe)
#number of good streamlines
num_streamlines = len(smallBundle_safe)


### mapping streamlines and initial weights (with a red bar)

hue = [0, 0]  # red only
saturation = [0.0, 1.0]  # black to white

lut_cmap = actor.colormap_lookup_table(
    scale_range=(0, 7),
    hue_range=hue,
    saturation_range=saturation)

weak_stream_actor = actor.line(weak_bundle, weak_ADI,
                               lookup_colormap=lut_cmap)
big_stream_actor = actor.line(big_bundle, big_ADI,
                            lookup_colormap=lut_cmap)
good_stream_actor = actor.line(good_bundle, good_ADI,
                           lookup_colormap=lut_cmap)

bar = actor.scalar_bar(lut_cmap, title = 'diameter')
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

### some usefull functions


#refreshing the ShowManager    
def refresh_showManager(i_ren, obj, slider):
    global show_m
    show_m.render()
    return

#function called by the graphe's slider
def add_graphe(i_ren, obj, slider):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, rectangles = ax.hist(x=smallADI_safe, bins=50, density=False, label = 'graphe')
    plt.xlabel('weight')
    plt.ylabel('number of streamlines')
    plt.title('histogram of iteration '+str(list_x_file[int(change_iteration_slider.value)]))
    plt.grid()
    plt.show()
    refresh_showManager(i_ren, obj, slider)
    return


#function to separate the streamlines in 3x2 arrayes according to the sliders
def refresh_3x2_arrays():
    global weak_ADI
    global weak_bundle
    global big_ADI
    global big_bundle
    global good_ADI
    global good_bundle

    big_bundle = []
    big_ADI = []
    good_bundle = []
    good_ADI = []
    weak_bundle = []
    weak_ADI = []

    i=0
    while(i<len(smallADI_safe)):
        if(smallADI_safe[i]<remove_small_weights_slider.value):
            weak_ADI.append(smallADI_safe[i])
            weak_bundle.append(smallBundle_safe[i])
        elif(smallADI_safe[i]>remove_big_weights_slider.value):
            big_ADI.append(smallADI_safe[i])
            big_bundle.append(smallBundle_safe[i])
        else:
            good_ADI.append(smallADI_safe[i])
            good_bundle.append(smallBundle_safe[i])
        i=i+1
    return

#function to refresh the 3 (weak, good, big) actors in the renderer
#using the 3x2 arrays
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
        if(len(good_ADI)>0):
            good_stream_actor = actor.line(good_bundle, good_ADI,
                              lookup_colormap=lut_cmap)
            renderer.add(good_stream_actor)
        if(len(weak_ADI)>0):
            weak_stream_actor = actor.line(weak_bundle, weak_ADI,
                              opacity =opacity_slider.value,
                              lookup_colormap=lut_cmap)
            renderer.add(weak_stream_actor)
        if(len(big_ADI)>0):
            big_stream_actor = actor.line(big_bundle, big_ADI,
                              opacity =opacity_slider.value,
                              lookup_colormap=lut_cmap)
            renderer.add(big_stream_actor)
    else:
        if(len(good_ADI)>0):
            good_stream_actor = actor.line(good_bundle)
            renderer.add(good_stream_actor)
        if(len(weak_ADI)>0):
            weak_stream_actor = actor.line(weak_bundle,
                              opacity =opacity_slider.value)
            renderer.add(weak_stream_actor)
        if(len(big_ADI)>0):
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
    global ADI
    global smallADI_safe
    
    #load the weights of the correct iteration according to the slider
    x_norm = np.load(args.weightsPath+'/'+list_x_file[int(slider.value)]+'.npy')
    
    num_ADI = np.zeros( nF )
    den_ADI = np.zeros( nF )

    for i in range(nIC):
        den_ADI = den_ADI + x_norm[i*nF:(i+1)*nF]
        num_ADI = num_ADI + x_norm[i*nF:(i+1)*nF] * Ra[i]
    
    ADI = 2 * ( num_ADI / ( den_ADI + np.spacing(1) ) ) * 1E6
    smallADI_safe = ADI[:num_computed_streamlines]
    
    
    refresh_3x2_arrays()
    refresh_3_stream_actors()
    #updating the number of good streamlines and the name of the iteration slider
    numbers_of_streamlines_in_interval.message = "Number of streamlines in interval: "+str(len(good_ADI))
    slider.text_template=list_x_file[int(slider.value)]
    refresh_showManager(i_ren, obj, slider)
    return

    
#function called by the remove_big_weight's slider
def remove_big_weight(i_ren, obj, slider):
    
    refresh_3x2_arrays()
    refresh_3_stream_actors()
    #updating the number of good streamlines
    numbers_of_streamlines_in_interval.message = "Number of streamlines in interval: "+str(len(good_ADI)) 
    refresh_showManager(i_ren, obj, slider)
    return

    
#function called by the remove_small_weight's slider
def remove_small_weight(i_ren, obj, slider):
    
    refresh_3x2_arrays() 
    refresh_3_stream_actors()
    #updating the number of good streamlines
    numbers_of_streamlines_in_interval.message = "Number of streamlines in interval: "+str(len(good_ADI))
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
        scale_range=(0, 7),
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


### adding sliders and renderer to a ShowManager

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
                                text_template="{value:.0f}",
                                length=40)

change_iteration_slider = ui.LineSlider2D(min_value=0,
				#we can't have max_value=num_iteration because
				#list_x_file[num_iteration] lead to an error
                                max_value=num_iteration-0.01,
                                initial_value=0,
                                text_template=list_x_file[0],
                                length=140)

remove_big_weights_slider = ui.LineSlider2D(min_value=0,
                                max_value=7,
                                initial_value=7,
                                text_template="{value:.2f}",
                                length=140)

remove_small_weights_slider = ui.LineSlider2D(min_value=0,
                                max_value=7,
                                initial_value=0,
                                text_template="{value:.2f}",
                                length=140)

opacity_slider = ui.LineSlider2D(min_value=0.0,
                                 max_value=1.0,
                                 initial_value=0.5,
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
opacity_slider_label.message = 'Opacity of small weights'

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
panel.add_element(color_slider, 'relative', (0.65, 0.9))
panel.add_element(numbers_of_streamlines_in_interval, 'relative', (0.05, 0.72))
panel.add_element(change_colormap_slider_label_weight, 'relative', (0.05, 0.59))
panel.add_element(change_colormap_slider_label_direction, 'relative', (0.5, 0.59))
panel.add_element(change_colormap_slider, 'relative', (0.4, 0.64))
panel.add_element(change_iteration_slider_label, 'relative', (0.05, 0.46))
panel.add_element(change_iteration_slider, 'relative', (0.65, 0.51))
panel.add_element(remove_big_weights_slider_label, 'relative', (0.05, 0.33))
panel.add_element(remove_big_weights_slider, 'relative', (0.65, 0.37))
panel.add_element(remove_small_weights_slider_label, 'relative', (0.05, 0.2))
panel.add_element(remove_small_weights_slider, 'relative', (0.65, 0.24))
panel.add_element(opacity_slider_label, 'relative', (0.05, 0.07))
panel.add_element(opacity_slider, 'relative', (0.65, 0.11))

panel.add_to_renderer(renderer)
renderer.reset_clipping_range()

show_m.render()
show_m.start()


if __name__ == "__main__":
	main()
