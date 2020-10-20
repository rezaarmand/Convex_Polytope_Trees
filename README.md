# Convex Polytope Trees

This repository contains an implementation of CPT
The code was tested for python 3.6


 


## How to use



You can use the following command to train CPT for the three depth paramters of 8, 9, and 10:


```
python run_polytop.py  --data MNIST  --depth  8 9 10  --lr 0.01  --epochs 70 --refine_epochs 50  -s 0.01  --initial_steepness 1 --polytope_sides  60  --refine
```
And for Connect4, depth 2

```
python run_polytop.py  --data connect4  --depth  2   --lr 0.01  --epochs 40 --refine_epochs 40  -s 0.01  --initial_steepness 1 --polytope_sides  40 --refine
```


For regression use the file `run_polytop_reg.py`and for binary classification run the file `run_polytop_binary_classification.py`.

The datasets are also provided.
