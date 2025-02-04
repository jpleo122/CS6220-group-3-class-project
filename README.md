# CS6220-group-3-class-project other components
- Colab Model Training ReadMe (Contains Code Package Requirements): https://docs.google.com/document/d/1XOQLcO3aaSrhjBngnoor1x7df0JBhkzD/edit
- GUI Repo: https://github.com/jpleo122/age-detector-saliency/tree/master/saliency
- Final report link: https://docs.google.com/presentation/d/1RWn3s7SzKMSHKJCnLVaogtSZTPfr4tB6alrIIgTP_Yw/edit?usp=sharing

## Overview

This specific repo runs age detection on faces within a given image using the other components. Once a face has been classified, a number of different saliency maps are computed for each face and saved locally. 

The saliency maps are generated by computing the pixel gradients of the input with respect to the probability that the face is a certain age range. These gradients are then transformed vis transformation functions defined in saliency.py. 

## How to run 

One can install the requirements in requirements.txt by running `python -m pip install -r requirements.txt`

Put input images in ./input folder 

Run  `python saliency_map.py`

See output in ./output

Output is formatted like so:

```
./output
    ->Images
    -> Predicted Image
    -> faces (one folder for each face)
        -> greyscale face
        -> transformations
            -> predicted age range saliency map
            -> other age range saliency maps
```

