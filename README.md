## Learning to Read and Follow Music in Complete Score Sheet Images

This repository contains the corresponding code for our paper

>[Henkel F.](https://www.jku.at/en/institute-of-computational-perception/about-us/people/florian-henkel/), 
>[Kelz R.](https://www.jku.at/en/institute-of-computational-perception/about-us/people/rainer-kelz/), and 
>[Widmer G.](https://www.jku.at/en/institute-of-computational-perception/about-us/people/gerhard-widmer/) <br>
"[Learning to Read and Follow Music in Complete Score Sheet Images]()".<br>
*In Proceedings of the 21st International Society for Music Information Retrieval Conference*, 2020


which is an extension of our previous work (which you can find by switching to the [*worms-2019*](https://github.com/CPJKU/audio_conditioned_unet/tree/worms-2019) branch):

>[Henkel F.](https://www.jku.at/en/institute-of-computational-perception/about-us/people/florian-henkel/),
>[Kelz R.](https://www.jku.at/en/institute-of-computational-perception/about-us/people/rainer-kelz/), and
>[Widmer G.](https://www.jku.at/en/institute-of-computational-perception/about-us/people/gerhard-widmer/) <br>
"[Audio-Conditioned U-Net for Position Estimation in Full Sheet Images](https://arxiv.org/pdf/1910.07254.pdf)". <br>
*In Proceedings of the 2nd International Workshop on Reading Music Systems*, 2019


### Data
The data used in this paper can be found [*here*](https://zenodo.org/record/3953657/files/msmd.zip?download=1) and should be placed in ```audio_conditioned_unet/data```. If you install the package
properly (see instructions below) this will be done automatically for you.

### Videos
In the folder [*videos*](https://github.com/CPJKU/audio_conditioned_unet/tree/master/videos) you will find several pieces from the test set, 
where our best performing model follows an incoming musical performance.

## Getting Started
If you want to try our code, please follow the instructions below.

### Setup and Requirements

First, clone the project from GitHub:

``` git clone https://github.com/CPJKU/audio_conditioned_unet.git```

In the cloned folder you will find an anaconda environment file which you should install using the following command:

``` conda env create -f environment.yml ```

Activate the environment:

``` conda activate audio_conditioned_unet```

Finally, install the project in the activated environment:

``` python setup.py develop ```

This last command will extract the sound font file and download the data.

### Software Synthesizer - FluidSynth

Make sure to have [FluidSynth](http://www.fluidsynth.org/) installed as we will use it to synthesize audio from MIDI. To this end we also provide you
 with a piano sound font which you can find in the folder [*audio_conditioned_unet/sound_fonts*](https://github.com/CPJKU/audio_conditioned_unet/tree/master/audio_conditioned_unet/sound_fonts)

### Check if everything works

To verify that everything is correctly set up, run the following command:

 ```python test_model.py --param_path ../models/CB_TA/best_model.pt --test_dir ../data/msmd/msmd_test --test_piece Anonymous__lanative__lanative_page_0 --config configs/msmd.yaml --plot```
 
This will run a pre-trained model on a test piece and plot the predictions on top of the score to the screen.
(Note: The '--plot' mode does not support audio playback. If you want this, you will need to create a video which will be explained below.)

## Training

If you want to train your own models, you will need to run *train_model.py*. This script can take several parameters
to change the network architecture and the training procedure. The most important parameters that you will need to set are
the paths to the train and validation set, in which blocks you would like to activate the [FiLM](https://arxiv.org/pdf/1709.07871.pdf) layers and which audio encoder you want to use.
You can also provide a log and a dump directory where the statistics during training and validation and the model parameters will be stored. 
The logs can be visualized by using [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html).

To give you an example, if you want to train a model with FiLM layers being activated in blocks B-H with a context based encoder and a LSTM, run the following command:

```python train_model.py --train_set ../data/msmd/msmd_train --val_set ../data/msmd/msmd_valid --config configs/msmd.yaml --film_layers 2 3 4 5 6 7 8 --audio_encoder CBEncoder --use_lstm```

To train the same models as in the paper you can check out the ```run_experiments.sh``` script.

## Evaluation
To reproduce the results shown in Table 2, we provide you with our trained models in the folder [*models*](https://github.com/CPJKU/audio_conditioned_unet/tree/master/models).
To evaluate a single model on the test set you need to run the following command:

```python eval_model.py --param_path ../models/<MODEL-FOLDER>/best_model.pt --test_dir ../data/msmd/msmd_test --config configs/msmd.yaml```

To get the results shown in Table 3 you have to add the ```--eval_onsets``` flag.

E.g., if you want to evaluate the context based model with tempo augmentation, you need to execute:

```python eval_model.py --param_path ../models/CB_TA/best_model.pt --test_dir ../data/msmd/msmd_test --config configs/msmd.yaml```

## Visualization

To see what our network actually does, we can create a video of its performance on a certain piece:

``` python test_model.py --param_path ../models/<MODEL-FOLDER>/best_model.pt --test_dir ../data/msmd/<TEST-DIR> --test_piece <PIECE> --config configs/msmd.yaml```

e.g.,  if you want to create a video for the test piece *Anonymous__lanative__lanative* using our best performing model,
 you need to execute:
 
```python test_model.py --param_path ../models/CB_TA/best_model.pt --test_dir ../data/msmd/msmd_test --test_piece Anonymous__lanative__lanative_page_0 --config configs/msmd.yaml```
 
 
 ## Acknowledgements
This project has received funding from the European Research Council (ERC) 
under the European Union's Horizon 2020 research and innovation program
(grant agreement number 670035, project "Con Espressione"). 

<img src="https://erc.europa.eu/sites/default/files/LOGO_ERC-FLAG_EU_.jpg" width="35%" height="35%">