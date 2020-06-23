## Textual Data Generation

This code implements our framework for generating examples for numerical reasoning over synthetic passages.  
Description of this framework is provided in sections 4.2 and A.2 in [our paper](https://arxiv.org/pdf/2004.04487.pdf).

To set up the environment, run the following commands, which install the required packages:
```bash
pip install -r requirements.txt
python -m spacy download en
```

To generate examples, run the `generate_examples.py` script (see the help menu for configuration options).

The code was tested in a **Python 3.6.8** environment.   

Contact: Mor Geva
