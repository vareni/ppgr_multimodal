# Multimodal Postprandial Glycemic Response Prediction

This repository contains the code accompanying the paper:

[**Predicting Postprandial Glycemic Response from Meal Images, Clinical Variables, and Gut Microbiome Information**](http://arxiv.org/abs/2609.24453)

Varvara Kondratyeva, Kamilia Zaripova, Nassir Navab, Azade Farshad  
Accepted at the [**MICCAI 2026 Workshop on Multimodal Learning with Medical Tabular Data (MultiTab)**](https://multitab-miccai-2026.github.io/).

The work investigates multimodal machine learning approaches for predicting postprandial glycemic response from meal images together with subject-specific clinical and gut microbiome information. Meal images are used to derive macronutrient representations that are combined with personalized tabular features for glucose-response prediction.

The project was originally developed as part of the master's thesis:

**Personalized Postprandial Glucose Prediction with Multimodal Machine Learning. Food Images, Clinical Data, and Microbiome Integration**  
Technical University of Munich, 2026.

## Project Overview

Postprandial glycemic response varies substantially between individuals, even after consuming similar meals. In this thesis, we investigate multimodal machine learning approaches for predicting individual glucose response from meal images and personalized tabular data.

The repository includes code for:

* preprocessing meal images and CGM data;
* computing glucose response targets such as AUC and iAUC;
* training image-based macronutrient prediction models;
* training downstream CGM response prediction heads;
* evaluating unimodal and multimodal models.

## Repository Structure

```text
CGM_multimodal/
├── DATA_PREPROCESSING/              # Data cleaning, preprocessing, and target preparation
├── EXPLORATION/                     # Exploratory analysis, sanity checks, and experimental notebooks
│
├── combined_model/                  # Main code for multimodal PPGR prediction
│   ├── models/                      # Neural network model definitions
│   │   ├── CGM_models.py            # CGM prediction heads
│   │   ├── joint_model.py           # Joint image + CGM model definition
│   │   └── myresnet.py              # ResNet-based image backbone / macronutrient model
│   │
│   ├── necks/                       # Intermediate feature projection and fusion modules
│   ├── utils/                       # Helper functions
│   │
│   ├── combined_model_macro_losses.py   # Main training script with macro and CGM losses
│   ├── combined_model.py                # Basic multimodal training script
│   └── mydataset.py                     # PyTorch dataset and data loading utilities
│
├── requirements.txt # Python dependencies 
├── requirements_dev.txt # Optional notebook dependencies 
├── README.md
├── .gitattributes
└── .gitignore
```

The main implementation is located in `combined_model/`. The `models/` directory contains the neural network components, including the ResNet-based macronutrient model adapted from the [RGB-DNet nutrition estimation framework](http://123.57.42.89/codes/RGB-DNet/nutrition.html) and the CGM prediction heads. The main training entry point is `combined_model_macro_losses.py`, which trains the combined model using both macronutrient-related losses and the downstream glucose response prediction loss.

The `DATA_PREPROCESSING/` directory contains scripts used to prepare the input data, including meal-level annotations, CGM-derived targets, and cleaned metadata (based on [CGMacros](https://github.com/PSI-TAMU/CGMacros)). The `EXPLORATION/` directory contains exploratory analyses and intermediate experiments that were used during model development but are not required for running the final training pipeline.


## Data and Preprocessing


The original dataset can be found and downloaded here: https://physionet.org/content/cgmacros/1.0.0/

The original preprocessing code is here: https://github.com/PSI-TAMU/CGMacros

All preprocessing scripts are located in:

```text
DATA_PREPROCESSING/
├── 0_parse_raw_data.py
├── 1_detect_wrapped.py
├── 2_filter_consumed.py
├── 3_manual_filter_data.py
├── 4_meal_times_filter.py
├── 5_add_microbiome.py
├── 6_split_train_val_test_to_txt.py
├── full_preprocessing.py
└── lunch_preprocessing.py
```

The preprocessing pipeline is organized as a sequence of numbered steps. The first step, `0_parse_raw_data.py` parses the raw input files and converts them into a cleaner intermediate format and is based on original preprocessing.

For convenience, the repository also includes wrapper scripts:

```bash
python DATA_PREPROCESSING/full_preprocessing.py
```

for running the complete preprocessing pipeline, and:

```bash
python DATA_PREPROCESSING/lunch_preprocessing.py
```

for preparing the lunch-only subset used in main experiments.

The resulting processed files are used by the dataset loader in:

```text
combined_model/mydataset.py
```

and are passed to the training scripts in `combined_model/`.

The expected processed data include meal-level image identifiers, macronutrient annotations, CGM-derived response targets, clinical variables, microbiome features, and train/validation/test split files.

## Environment Setup

Create a new conda environment:

```bash
conda create -n ppgr-thesis python=3.10
conda activate ppgr-thesis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, make sure that the installed PyTorch version is compatible with the available CUDA version.

## Preprocessing

The preprocessing pipeline can be run on CPU and does not require GPU resources. For the experiments in this repository, the lunch subset is prepared using the following script:

```bash
python DATA_PREPROCESSING/lunch_preprocessing.py
```


The individual preprocessing steps are also available in `DATA_PREPROCESSING/` as numbered scripts, but they do not need to be executed manually for the standard lunch-based experiments.


## Training

Model training is performed from the `combined_model/` directory. The standard training command used for the main experiments is:

```bash
cd combined_model

python -u combined_model_macro_losses.py \
    --rgbd \
    --glucose \
    --cgm_model 'CGMHead' \
    --train_macro
```

This command trains the combined model using image input, glucose-response prediction, and the default MLP-based CGM head (`CGMHead`). The `--train_macro` flag enables training of the macronutrient prediction branch together with the CGM prediction task.

ATTENTION: current `combined_model_macro_losses.py` will train two models by design: furst without microbiome features, then with them.

The main available training arguments are:

```text
--run_name              Name of the experiment run. If not provided, the current date and time are used.
--glucose               Include glucose-related clinical data.
--microbiome            Add microbiome features to the CGM prediction model.
--model                 Image backbone model name. Default: resnet101.
--rgbd                  Legacy flag inherited from the repository on which the image model was originally based; it is retained for compatibility but is redundant in the current implementation.
--resume                Resume training from a saved checkpoint (provide path/to/checkpoint.pth).
--cgm_model             CGM prediction head to use. Default: CGMHead.
--train_macro           Train the macronutrient prediction branch.
--use_latent_macros     Use latent macronutrient representation.
--latent_macro_dim      Dimensionality of the latent macronutrient representation. Default: 4.
--macro_loss_weight     Weight of the macronutrient loss in the combined objective. Default: 0.5.
```


The CGM prediction head can be changed using the `--cgm_model` argument. For example:

```bash
python -u combined_model_macro_losses.py \
    --rgbd \
    --glucose \
    --cgm_model 'CGMHeadAttentionMicroFiLM' \
    --train_macro
```

Training requires GPU resources. Preprocessing can be performed separately on CPU before launching model training.


## Model Variants

The repository supports several CGM prediction heads:

* `CGMHead`: multilayer perceptron baseline;
* `CGMHeadAttention`: attention-based feature fusion (legacy);
* `CGMHeadMicroFiLM`: microbiome-conditioned FiLM modulation (legacy);
* `CGMHeadAttentionMicroFiLM`: attention-based fusion with microbiome conditioning.


## Publication

This repository accompanies the paper:

**Predicting Postprandial Glycemic Response from Meal Images, Clinical Variables, and Gut Microbiome Information**

Varvara Kondratyeva, Kamilia Zaripova, Nassir Navab, Azade Farshad

Accepted at the **MICCAI 2026 Workshop on Multimodal Learning with Medical Tabular Data (MultiTab)**.

📄 **Preprint:** [arXiv](http://arxiv.org/abs/2609.24453)

The paper will appear in the **Lecture Notes in Computer Science (LNCS)** proceedings published by Springer Nature. The Springer link and DOI will be added here once available.

The arXiv version corresponds to the pre-peer-review submitted manuscript.

## Thesis

This repository accompanies the master’s thesis:

**Personalized Postprandial Glucose Prediction with Multimodal Machine Learning. Food Images, Clinical Data, and Microbiome Integration**

Technical University of Munich, 2026,
Chair for Computer-Aided Medical Procedures and Augmented Reality

## Citation


If you use this repository, please cite our paper:

```bibtex
@article{kondratyeva2026ppgr,
  title   = {Predicting Postprandial Glycemic Response from Meal Images,
             Clinical Variables, and Gut Microbiome Information},
  author  = {Kondratyeva, Varvara and Zaripova, Kamilia and
             Navab, Nassir and Farshad, Azade},
  year    = {2026},
  journal = {arXiv preprint arXiv:2609.24453}
}
```

## License

No open-source license has been assigned to this repository yet.

The code is made publicly available for research transparency and reproducibility. Parts of the image-based nutrition estimation code are adapted from RGB-DNet, and parts of the data preprocessing pipeline are based on the CGMacros repository; those components remain subject to the terms of their respective original licenses.

The accompanying paper/preprint is distributed separately under the license specified on arXiv.
## Contact

For questions about the repository, please contact:

Varvara Kondratyeva,
varvara.kondratyeva@tum.de

Technical University of Munich,
Chair for Computer-Aided Medical Procedures and Augmented Reality
