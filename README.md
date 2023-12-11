## child-mind-institute-detect-sleep-states: 3rd place solution code

It's 3rd place solution to Kaggle competition: child-mind-institute-detect-sleep-states: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states

This repo contains the code for training the models, while the solution writeup is available here: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459599

### Environment

To setup the environment:
* Install python
* Install `requirements.txt` in the fresh python environment

## LB solution

### Training

* Download raw data from https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data and extract it to `./data/raw_data` directory
* To generate static_features run `create_static_features.py`
* To generate preprocess information run `./preprocessing/runs_prep.py`
* To train all models, run: `./runs.py`

### Inference

Final inference kernel is available here: https://www.kaggle.com/code/trasibulo/final-inference/notebook

