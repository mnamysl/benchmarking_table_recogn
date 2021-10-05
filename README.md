# Table Recognition Benchmarking

This repository contains the code that can be used to reproduce the results presented in our paper: TBA. 

## Quick Start

### Prerequisites

1. Install the [ICU](https://icu.unicode.org/home) library. For the Ubuntu OS, you can use the following command:

```bash
sudo apt-get install libicu-dev icu-devtools
```
2. Install dependencies to the python packages as follows:

```bash
pip3 install -r requirements.txt
```

3. Extract the ground-truth data and the recognition results:

```bash
tar -xvzf gt.tar.gz gt
tar -xvzf res.tar.gz res
```

### Using the Code

#### Command-Line Arguments:
```bash
usage: eval_pubmed.py [-h] --res RES_DIR [--gt GT_DIR] [--single_variant]
                      [--method {ICDAR,Abbyy,TabulaJson,Unknown}]

optional arguments:
  -h, --help            show this help message and exit
  --res RES_DIR         a directory wih recognition results (default: )
  --gt GT_DIR           a directory wih ground-truth annotations (default: gt)
  --single_variant      indicates whether to perform single-variant evaluation (default: True)
  --method {ICDAR,Abbyy,TabulaJson,Unknown}
                        parsing method (icdar, abbyy, tabula-json) (default: icdar)
```
#### Running the Evaluation Script:

Use the following commands to reproduce the multi-variant evaluation:

```bash
python3 eval_pubmed.py --res res/tab.iais
python3 eval_pubmed.py --res res/tabby
python3 eval_pubmed.py --res res/abbyy --method abby
python3 eval_pubmed.py --res res/tabula_stream --method tabula.json
python3 eval_pubmed.py --res res/tabula_lattice --method tabula.json
```
Note that you need to specify the method used to parse the recognition results for each method (using the ```--method``` switch). [The ICDAR 2013 Table Competition](https://www.tamirhassan.com/html/competition/dataset-format.html#structure-model) format is used by default.

Add the ```--single_variant``` switch to perform single-variant evaluation, e.g.:

```bash
python3 eval_pubmed.py --res res/tab.iais --single_variant
```

#### Evaluating Other Table Recognition Methods

The PDF files from our biomedical data set can be found [here](https://zenodo.org/record/5549977#.YVxrS3uxVH6).

### Citing Our Work

Please cite our paper when using the code:
```
TBA
```

### Authors

* [Marcin Namysl](https://www.researchgate.net/profile/Marcin-Namysl-2)

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.