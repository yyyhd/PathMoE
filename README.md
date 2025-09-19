# PathMoE

Neoadjuvant therapy efficacy assessment is critical for prognostic evaluation and clinical decision-making in locally advanced breast cancer. This study developed an AI-based pathological multistain mixture-of-expert model, PathMoE, to predict pathological complete response (pCR) to Neoadjuvant chemotherapy (NAC) using pre-treatment breast biopsy whole slide images (WSIs) from multistain (H\&E, ER, PR, HER2, and Ki-67). Leveraging Virchow2, a large-scale pathology foundation model, we extracted generalizable WSI representations and integrated them through a multi-instance, expert-gated fusion strategy. PathMoE was trained and tested on 12,150 WSIs from 2,430 patients across five medical centers. Across independent external cohorts and a prospective cohort, analyses consistently demonstrated that PathMoE achieved robust pCR prediction, with added potential to predict response to combined neoadjuvant chemoimmunotherapy (NACI). PathMoE enabled robust risk stratification and served as an independent prognostic indicator. Model attention maps highlighted anti-tumor immune cell patterns as key decision drivers, while RNA sequencing (RNA-seq) linked high PathMoE scores to immune-related pathways. Collectively, these findings suggest that PathMoE is a promising tool for predicting pCR to NAC  and personalized management in breast cancer. 

<img src="main.png" width="800px" align="center" />

## Installation

First clone the repo and cd into the directory:
```shell
git clone https://github.com/yyyhd/PathMoE
cd PathMoE
```

Create a new enviroment with anaconda.
```shell
conda create -n PathMoE python=3.10 -y --no-default-packages
conda activate PathMoE
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
## Model Download

The MUSK models can be accessed from [HuggingFace Hub](https://huggingface.co/zzhuo-cs/PathMoE/tree/main/pytorch_model.pt).

You need to agree to the terms to access the models and login with your HuggingFace write token:
```python
from huggingface_hub import login
login(<huggingface write token>)
```


## evaluation indicator
```
python eval.py
```

## Acknowledgements
The project was built on amazing open-source repositories: CLAM. We thank the authors and developers for their contributions.


## Issues
Please open new threads or address questions to maoning@pku.edu.cn or sen.yang.scu@gmail.com

## License
This model may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the MUSK model and its derivatives is prohibited and requires prior approval.
