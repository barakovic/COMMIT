# How to use COMMIT visualization tool

This tutorial explains how to execute and use an interactive visualization tool to study convergence of COMMIT's microstructure estimates.

## execute the script

First, download the following [script](https://github.com/LorisPilotto/COMMIT/tree/master/doc/tutorials/visualizationTool/script.py).
Before executing the script, make sure you have a valide outpute of COMMIT.
Then, on the terminal go on the directory where the script is and type the command:

```
python /home/.../CommitOutput/ 1000
```
which will launch the program and load 1000 streamlines.


If you have both Cylinder and Stick commit's output, the terminal will show something like this:
```
/home/pilotto/Packages/anaconda2/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Which model do you want to load (1 for 'Cylinder', 2 for 'Stick') : 
```
Select the model you want and press enter.

You should see something like this:
![launch](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/launch.png)

## how to use the visualization tool


![iteration](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/iteration.png)
Once on the program, you can move the sliders. The most important one is the slider, showed by the pink arrow, that goes throught the iterations. The number of the iteration you are visualizing appeare below this slider.

![treshold](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/treshold.png)
The streamlines with a weight inside the pink box are the streamlines you find important. You can reduce the opacity of the streamlines with a weight outside the pink box with the lower slider (as showed on the image).
You can also notice that "Number of streamlines in interval" alows you to know how much streamlines have their weight's value inside the pink box.

![colorblue](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/colorblue.png)
By moving the upper slider you can change the color of the streamlines

![direction](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/direction.png)
By moving the slider showed by the pink arrow you can change the color map. Putting the slider on the left allows you to see a mapping between streamlines and their weight. And by puting the slider on the right you can see the streamlines with a directionally-encoded color (other functionalities are still available).

![butons](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/butons.png)
Finaly there are two butons, one to save the current view in the folder of the script and one to show a graph of the weights.
