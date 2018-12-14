# How to use COMMIT visualization tool

This tutorial explains how to execute and use an interactive visualization tool to study convergence of COMMIT's microstructure estimates.

## execute the script

First, download the following [script](https://github.com/LorisPilotto/COMMIT/tree/master/doc/tutorials/visualizationTool/script.py).
Then, on the terminal go on the directory where the script is and type the command:

```
python /home/.../CommitOutput/ 1000
```
which will launch the program.


If you have both Cylinder and Stick commit's output, the terminal will show something like this:
```
/home/pilotto/Packages/anaconda2/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Which model do you want to load (1 for 'Cylinder', 2 for 'Stick') : 
```
Select the model you want and press enter.

You should see something like this:
![launch](https://github.com/LorisPilotto/COMMIT/doc/tutorials/visualizationTool/launch.png)

## how to use the visualization tool


