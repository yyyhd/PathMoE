# Multistain pathology AI predicts therapy response and prognosis in breast cancer

Accurate prediction of response to neoadjuvant chemotherapy (NAC) is crucial for optimizing treatment and improving outcomes in breast cancer. Here, we present PathMoE, a multistain mixture-of-experts framework that jointly models paired hematoxylin-eosin and immunohistochemistry whole-slide images (WSIs, including ER, PR, HER2, and Ki-67) to predict pathological complete response (pCR) and patient prognosis. PathMoE leverages a pathology foundation model and stain-specific expert network, in which an adaptive gating mechanism dynamically integrates heterogeneous morphological and molecular features to yield biologically informed predictions. PathMoE was developed and validated using 12,150 multistain WSIs from 2,430 patients across five medical centers, including three independent external cohorts, a prospective validation cohort, and a neoadjuvant chemoimmunotherapy cohort. Across datasets and molecular subtypes, PathMoE outperformed standard clinicopathological variables and molecular biomarkers, improving pCR prediction accuracy by up to 22\% and delivering over 28\% gains in predicting response to neoadjuvant chemoimmunotherapy.
## Installation

First clone the repo and cd into the directory:

```bash
git clone https://github.com/yyyhd/PathMoE
cd PathMoE-main
```
Create a new enviroment with anaconda.

```shell
conda create -n PathMoE python=3.8 -y --no-default-packages
conda activate ParhMoE
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Model Download

The PathMoE models can be accessed from [HuggingFace Hub]

## Image Processing Pipeline

### Extract Tiles from Whole Slide Images
Preprocess the slides following [CLAM](https://github.com/mahmoodlab/CLAM), including foreground tissue segmentation and stitching. 

### Extract Image Feature Embeddings
 Download the pretrained [Virchow2 model weights](https://huggingface.co/paige-ai/Virchow2), put it to *./weights/* and load the model



## Evaluation 

To reproduce the results in our paper, we provide a reproducible result on JCH cohort.

* First download our processed JCH cohort frozen features [here](https://pan.baidu.com/s/1tp4jWYuN7SO3oodCM7XbhA?pwd=6piu)
* Put the extracted features to *./features/* 
* Run the following command:
```shell
python eval.py
```
The test_error and auc will be printed to the screen. 
```python
test_error:  0.1981981981981982   auc:  0.8567470664928292
```

The computed scores for this cohort will be stored at `eval_results/EVAL_first`

## Acknowledgements
The project was built on many amazing repositories: [Virchow2](https://huggingface.co/paige-ai/Virchow2), [CLAM](https://github.com/mahmoodlab/CLAM). We thank the authors and developers for their contributions.

## Issue
Please open new threads or address questions to maoning@pku.edu.cn or sen.yang.scu@gmail.com

## License

PathMoE is made available under the CC BY-NC-SA 4.0 License and is available for non-commercial academic purposes.


