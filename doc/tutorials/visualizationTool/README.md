# How to use visEst, COMMIT estimates visualization tool

This tutorial explains how to execute and use the interactive visualization tool. The tool allows to study convergence of COMMIT's microstructure estimates.

## Execute the script

First, download the following [script](https://github.com/LorisPilotto/COMMIT/tree/master/doc/tutorials/visualizationTool/visualizationTool.py).

Before executing the script, make sure you have a valid output of COMMIT.

Then, on the terminal go on the directory where the script is and type the command:

```
python script.py /home/.../CommitOutput/ 1000
```
which will launch the program and load 1000 streamlines.


If you have both CylinderZeppelinBall and StickZeppelinBall models, you will have the possibility to choose your model:

```
Which model do you want to load (1 for 'Cylinder', 2 for 'Stick') : 
```

Type 1 or 2 in order to select the model and press enter.

You should see a visualization similar to:
![launch](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/launch.png)

## How to use the visualization tool

- **Change the iteration of COMMIT we want to visualize**
![iteration](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/iteration.png)
Once on the program, you can move the sliders, showed by the pink arrow, which goes through the iterations previously saved. The number of the iteration you are visualizing appears below this slider.

- **A lower and a upper threshold to lower the visibility of unwanted weights**
![treshold](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/treshold.png)
The streamlines with a weight inside the pink box are the streamlines you find important. You can reduce the opacity of the streamlines with a weight outside the pink box with the lower slider (as showed on the image).
You can also notice that "Number of streamlines in interval".

- **Changing the colormap of the streamlines**
![color](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/color.png)
By moving the upper slider you can change the color of the streamlines.

- **Visualize the streamlines with the directionally-encoded color**
![direction](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/direction.png)
By moving the slider showed by the arrow you can change the colormap. Putting the slider on the left allows you to see a mapping between streamlines and their weights. By setting the slider on the right you can see the streamlines with a directionally-encoded color (other functionalities are still available).

- **Button to save the image and to show a graph of the weights**
![butons](https://github.com/LorisPilotto/COMMIT/blob/pilotto_project/doc/tutorials/visualizationTool/butons.png)
Finally there are two buttons, one to save the current view in the folder of the script and one to show a graph of the weights.
