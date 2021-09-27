# TreeCodeGen

This repo contains the code for the Structured model for CoNaLa dataset, as described in the paper: https://aclanthology.org/2021.findings-acl.384.pdf


## Instructions:

1. Download the data from http://conala-corpus.github.io
1. Build a file with NL intents of training set: Refer to datasets/conala/retrive_src.py. The file will be saved as src.txt in data/conala
1. Parse those NL intents and build a vocabulary: refer to https://github.com/nxphi47/tree_transformer for more details on setting up the parser and run convert_ln.py
1. Build train/dev/test dataset: Run datasets/conala/dataset_hie.py
1. Train the model: scripts/conala/train.sh
1. Test the model: scripts/conala/test.sh


This repo is built on top of two projects:
a. TRANX: https://github.com/pcyin/tranX
b. tree_transformer: https://github.com/nxphi47/tree_transformer

So please cite their work. 

## Reference:
```

@article{dahalanalysis,
  title={Analysis of Tree-Structured Architectures for Code Generation},
  author={Dahal, Samip and Maharana, Adyasha and Bansal, Mohit}
}

```