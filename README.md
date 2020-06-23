## Injecting Numerical Reasoning Skills into Language Models

This repository contains the accompanying code for the paper:

**"Injecting Numerical Reasoning Skills into Language Models."** Mor Geva*, Ankit Gupta* and Jonathan Berant. *In ACL, 2020*.
[[PDF]](https://arxiv.org/pdf/2004.04487.pdf)


### Structure
The repository contains:
* Implementation/pre-training/finetuning of GenBERT on MLM/synthetic-data/DROP/SQuAD (in `pre_training` dir)
* Code and vocabularies for textual data generation (in `textual_data_generation` dir)
* Code for numerical data generation (in `pre_training/numeric_data_generation` dir)   

Instructions for downloading our data + models are in the README of `pre_training` dir.

---
### Citation
```
@inproceedings{ggb2020injecting,
  title={Injecting Numerical Reasoning Skills into Language Models},
  author={Geva, Mor and Gupta, Ankit and Berant, Jonathan},
  booktitle={ ACL },
  year={ 2020 }
}
```
