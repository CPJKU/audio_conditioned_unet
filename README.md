
# Audio-Conditioned U-Net for Position Estimation in Full Sheet Images 

This repository contains the code for the paper
 [*Audio-Conditioned U-Net for Position Estimation in Full Sheet Images*](https://arxiv.org/pdf/1910.07254.pdf).

### Data
The data used in this paper can be found in the folder [*data*](https://github.com/CPJKU/audio_conditioned_unet/tree/master/data)

### Videos
In the folder [*videos*](https://github.com/CPJKU/audio_conditioned_unet/tree/master/videos) you will find several pieces from the test set, where our model predicts matching 
positions in the score.

## Getting Started
If you want to try our code, please follow the instructions below.

### Setup and Requirements

First, clone the project from GitHub:

``` git clone https://github.com/CPJKU/audio_conditioned_unet.git```

In the cloned folder you will find an anaconda environment file which you should install using the following command:

``` conda env create -f environment.yml ```

Activate the environment:

``` conda activate audio_conditioned_unet```

Finally, install the project in the activate environment:

``` python setup.py develop ```

This last command will extract the sound font file and the data.

### Software Synthesizer - FluidSynth

Make sure to have [FluidSynth](http://www.fluidsynth.org/) installed as we will use it to synthesize audio from MIDI. To this end we also provide you
 with a piano sound font which you can find in the folder [*audio_conditioned_unet/sound_fonts*](https://github.com/CPJKU/audio_conditioned_unet/tree/master/audio_conditioned_unet/sound_fonts)

### Check if everything works

To verify that everything is correctly set up, run the following command:

 ``` python test_score_unet.py --param_path ../models/film_layer_C-G/best_model.pt --test_piece ../data/nottingham_test/ashover41 --plot```
 
This will run a pre-trained model on a test piece and plot the predictions and the corresponding ground truth to the screen.
(Note: The '--plot' mode does not support audio playback. If you want this, you will need to create a video which will be explained below.)

## Training

If you want to train your own models, you will need to run *worms_experiment.py*. This script can take several parameters
to change the network architecture and the training procedure. The most important parameters that you will need to set are
the paths to the train and validation set and in which blocks you would like to activate the [FiLM](https://arxiv.org/pdf/1709.07871.pdf) layers.
You can also provide a log and a dump directory where the statistics during training and validation and the model parameters will be stored. 
The logs can be visualized by using [Tensorboard](https://github.com/lanpa/tensorboardX)
(as of the latest Pytorch version Tensorboard is now included in Pytorch, which we did not yet adapt in our code).

To give you an example, if you want to train a model with FiLM layers being activated in blocks C-G, run the following command:

``` python worms_experiment.py --train_set ../data/nottingham_train --val_set ../data/nottingham_valid --film3 --film4 --film5 --film6 --film7```

## Evaluation
To reproduce the results shown in Table II, we provide you with our trained models in the folder [*models*](https://github.com/CPJKU/audio_conditioned_unet/tree/master/models).
To evaluate a single model on the test set you need to run the following command:

``` python worms_evaluation.py --param_path ../models/<MODEL-FOLDER>/best_model.pt --test_set ../data/nottingham_test```

e.g., if you want to evaluate the model with active FiLM layers in block C-G, you need to execute:

``` python worms_evaluation.py --param_path ../models/film_layer_C-G/best_model.pt --test_set ../data/nottingham_test```

## Visualization

To see what our network actually does, we can create a video of its performance on a certain piece:

``` python test_score_unet.py --param_path ../models/<MODEL-FOLDER>/best_model.pt --test_piece ../data/nottingham_test/<PIECE> ```

e.g.,  if you want to create a video for the test piece *ashover41* for the model with active FiLM layers in block C-G,
 you need to execute:
 
 ``` python test_score_unet.py --param_path ../models/film_layer_C-G/best_model.pt --test_piece ../data/nottingham_test/ashover41 ```