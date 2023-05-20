# BERTUS-ABSC

BERTUS Aspect-Based Sentiment Classification (ABSC) using LCR-Rot-hop++.

Van Berkum, Van Megen, Savelkoul, Weterman, and Frasincar ([2021](https://doi.org/10.1145/3486622.3494003)) describes the official implementation of the methods.

## Installation
- Set up a virtual environment:
    - Download and install Python 3.6.2 (https://www.python.org/downloads/) and Anaconda (https://www.anaconda.com/products/individual).
    - Create an Anaconda virtual environment using Python 3.6.2.
    - Install the required packages to the virtual environment by running this command on the Anaconda Prompt screen:
      ```pip install -r requirements.txt```.
      - make sure to place the requirements.txt in the user directory.
    - During the installation, Visual Studio C++ build tools error may occur. If so, download and install the Visual Studio Community 2017: desktop development with C++.
    - If you want to use GPU to train and test the model, please also download and install NVIDIA CUDA Visual Studio Integration 9.0.
    - You can open and edit the code in any editor, we used the PyCharm IDE: https://www.jetbrains.com/pycharm/.

## User Guide
- Get raw data for every domain separately.
    - Run raw_data.py for the MP3 player (Creative), DVD player (Apex), hotel (hotel), and cell phone (Nokia) domains.
- Get BERT word embeddings: converting each word into a 768-dimensional vector 
    - Run Jupyter notebook file in the getBERT folder *on Google Colab* to obtain BERT embeddings (see the file for
      the detailed explanations).
- Data preprocessing before running the code:
    - Run prepare_bert.py and prepare_bert_split.py for every domain of interest.
- If you use other datasets, tune the hyperparameters:
  - run main_hyper.py for LCR-Rot-hop++ hyperparameters.
  - run main_hyper_BERTUS.py for BERTUS hyperparameters.
  - If you are using the same datasets, you can use pre-set values in main_test.py.
- Run main_test.py to train and test the BERTUS-LCR-Rot-hop++ model.
  - Make sure to set writable = 1 in config.py if you want to save the results.

## References

This code is partially adapted from the following papers.

1. Lee, Frasincar, and Trusca
https://github.com/ejoone/DIWS-ABSC/tree/main

Junhee Lee, Flavius Frasincar, and Maria Mihaela Trusca. 
A cross-domain aspect-based sentiment classification by masking the domain-specific words. 
In 38th ACM/SIGAPP Symposium on Applied Computing (SAC 2023). ACM, 2023.


2. Trusca, Wassenberg, Frasincar, and Dekker (2020).
https://github.com/mtrusca/HAABSA_PLUS_PLUS

Trusca, M. M., Wassenberg, D., Frasincar, F., and Dekker, R. (2020). A hybrid approach
for aspect-based sentiment analysis using deep contextual word embeddings and hierarchical
attention. In 20th International Conference of Web Engineering (ICWE 2020), volume 12128
of LNCS, pages 365–380. Springer.

3. van Berkum, van Megen, Savelkoul, Weterman, and Frasincar (2021).
https://github.com/stefanvanberkum/CD-ABSC

van Berkum, S., van Megen, S., Savelkoul, M., Weterman, P., and Frasincar, F. (2021). Fine-tuning for cross-domain aspect-based sentiment classification. In IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT 2021), page
524–531. ACL.
