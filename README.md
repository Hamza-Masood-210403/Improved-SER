# Compact Graph Architecture for Speech Emotion Recognition

## Dependency installation

The testing environment can be created by importing _environment.yml_ to conda or by using the _requirements.txt_ file


## Preprocessing Data

The process for IEMOCAP database is in preprocess directory. The process converts the database into one txt file including graph structure and node attributes.

Note: you can download the processed data from [here](https://drive.google.com/file/d/1_3H_wByS-cSLLG7vrhgfvdzCnjaXJ2ui/view?usp=sharing) and put in this directory if not already present:

```
/dataset/
  IEMOCAP/
    IEMOCAP.txt
```

## Training

You can train the model with running main.py . 


```
usage: python main.py

optional arguments:
  -h, --help            Show this help message and exit
  -device               Which gpu to use if any
  -batch_size           Input batch size for training
  -iters_per_epoch      Number of iterations per each epoch
  -epochs               Number of epochs to train
  -lr                   Learning rate
  --num_layers          Number of inception layers
  --hidden_dim          Number of hidden units for MLP
  --final_dropout       Dropout for classifier layer
  --Normalize           Normalizing data
  --patience            Patience for early stopping
  --graph_type          To set the graph type (line or cycle)
  --graph_pooling_type  Pooling type (min, max or sum)
```

<br><br><br>
