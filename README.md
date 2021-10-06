# Table Recognition Benchmarking

This repository contains the code that can be used to reproduce the results presented in our paper: **Benchmarking Table Recognition Performance on Biomedical Literature on Neurological Disorders**. 

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
usage: eval_pubmed.py [-h] --res RES_DIR [--gt GT_DIR] [--single_variant] [--method {ICDAR,Abbyy,TabulaJson,Unknown}] 
  [--complexity [{0,1,2} [{0,1,2} ...]]] [--verbose] [--ignore_fp]

optional arguments:
  -h, --help            show this help message and exit
  --res RES_DIR         a directory wih recognition results (default: )
  --gt GT_DIR           a directory wih ground-truth annotations (default: gt)
  --single_variant      indicates whether to perform single-variant evaluation (default: True)
  --method {ICDAR,Abbyy,TabulaJson,Unknown}
                        parsing method (icdar, abbyy, tabula-json) (default: icdar)
  --complexity [{0,1,2} [{0,1,2} ...]]
                        table complexity level (0=simple, 1=complicated, 2=complex) (default: [0, 1, 2])
  --verbose             print verbose info (default: False)
  --ignore_fp           ignore all false-positively recognized tables (default: False)
```
#### Running the Evaluation Script:

In this section, we present the steps that need to be performed to reproduce the result from our experiments.

* The multi-variant evaluation is performed by default. To perform single-variant evaluation you need to add the ```--single_variant``` switch to the call.

* Note that you need to specify the method used to parse the recognition results for each method (using the ```--method``` switch; [the ICDAR 2013 Table Competition](https://www.tamirhassan.com/html/competition/dataset-format.html#structure-model) format is used by default). 

* The script first loads the ground-truth annotations and the recognition results. Subsequently, the evaluation procedure is performed. The last line printed to the output contains the final evaluation scores. 

In the following, we present the exact commands used to trigger the evaluation process and the expected final scores for each method.

* [ABBYY  Fine  Reader  Engine](https://www.abbyy.com/ocr-sdk) (SDK v12)
```bash
python3 eval_pubmed.py --res res/abbyy --method abbyy
```
```FINAL RESULT: TP:120661 FN:77521 FP:78442 GT=198182 RES=199103 PRECISION=0.6060 RECALL=0.6088 F1=0.6074 F0.5=0.6066```

```bash
python3 eval_pubmed.py --res res/abbyy --method abbyy --single_variant
```
```FINAL RESULT: TP:117904 FN:80278 FP:78903 GT=198182 RES=196807 PRECISION=0.5991 RECALL=0.5949 F1=0.5970 F0.5=0.5982```

* [TAB.IAIS](https://arxiv.org/abs/2105.11879) method:
```bash
python3 eval_pubmed.py --res res/tab.iais
```
```FINAL RESULT: TP:95199 FN:102983 FP:49029 GT=198182 RES=144228 PRECISION=0.6601 RECALL=0.4804 F1=0.5561 F0.5=0.6141```

```bash
python3 eval_pubmed.py --res res/tab.iais --single_variant
```
```FINAL RESULT: TP:92666 FN:105516 FP:46169 GT=198182 RES=138835 PRECISION=0.6675 RECALL=0.4676 F1=0.5499 F0.5=0.6149```

* [TabbyPDF](https://github.com/cellsrg/tabbypdf)
```bash
python3 eval_pubmed.py --res res/tabby
```
```FINAL RESULT: TP:111195 FN:86987 FP:95381 GT=198182 RES=206576 PRECISION=0.5383 RECALL=0.5611 F1=0.5494 F0.5=0.5427```

```bash
python3 eval_pubmed.py --res res/tabby --single_variant
```
```FINAL RESULT: TP:108207 FN:89975 FP:107333 GT=198182 RES=215540 PRECISION=0.5020 RECALL=0.5460 F1=0.5231 F0.5=0.5102```

* [Tabula](https://github.com/tabulapdf/tabula-java) (v1.0.4):
  * 'Stream' mode:
  ```bash
  python3 eval_pubmed.py --res res/tabula_stream --method tabula-json
  ```
  ```FINAL RESULT: TP:71693 FN:126489 FP:610179 GT=198182 RES=681872 PRECISION=0.1051 RECALL=0.3618 F1=0.1629 F0.5=0.1225```

  ```bash
  python3 eval_pubmed.py --res res/tabula_stream --method tabula-json --single_variant
  ```
  ```FINAL RESULT: TP:69435 FN:128747 FP:624288 GT=198182 RES=693723 PRECISION=0.1001 RECALL=0.3504 F1=0.1557 F0.5=0.1168```

  * 'Lattice' mode
  ```bash
  python3 eval_pubmed.py --res res/tabula_lattice --method tabula-json
  ```
  ```FINAL RESULT: TP:26086 FN:172096 FP:21613 GT=198182 RES=47699 PRECISION=0.5469 RECALL=0.1316 F1=0.2122 F0.5=0.3353```

  ```bash
  python3 eval_pubmed.py --res res/tabula_lattice --method tabula-json --single_variant
  ```
  ```FINAL RESULT: TP:25689 FN:172493 FP:15065 GT=198182 RES=40754 PRECISION=0.6303 RECALL=0.1296 F1=0.2150 F0.5=0.3556```

#### Evaluating Other Table Recognition Methods

The results of other methods can be easily evaluated. You only need to process the PDF files from [our biomedical data set](https://zenodo.org/record/5549977#.YVxrS3uxVH6) using your table recognition method and store the results in one of the supported output formats (e.g., in the ICDAR 2013 Table Competition format). Then you can to call the evaluation script with the path to your results as follows:

```bash
python3 eval_pubmed.py --res <path_to_your_result>
```

#### Subset-Level Evaluation

To perform the evaluation at the subset-level, you need to specify an additional ```--complexity``` argument, which represents the table complexity classes (0 = simple, 1 = complicated, 2 = complex). 

* In addition, the ```--ignore_fp``` flag can be used to skip all false-positively detected tables.

* Note that the subset-level experiment was performed using the single-variant evaluation mode.

The following example demonstrates how to evaluate the Tab.IAIS method on the full benchmark without the false-positively detected tables:

```bash
python3 eval_pubmed.py --res res/tab.iais --complexity 0 1 2 --ignore_fp --single_variant
```
```FINAL RESULT: TP:92666 FN:105516 FP:30615 GT=198182 RES=123281 PRECISION=0.7517 RECALL=0.4676 F1=0.5765 F0.5=0.6702```

Another example shows how to evaluate the Tabby method using the subset containing simple tables:

```bash
python3 eval_pubmed.py --res res/tabby --complexity 0 --ignore_fp --single_variant
```
```FINAL RESULT: TP:62592 FN:45936 FP:28302 GT=108528 RES=90894 PRECISION=0.6886 RECALL=0.5767 F1=0.6277 F0.5=0.6629```

### Citing Our Work

Please cite our paper when using the code:
```
TBA
```

### Authors

* [Marcin Namysl](https://www.researchgate.net/profile/Marcin-Namysl-2) [[ORCID]](https://orcid.org/0000-0001-7066-1726)
* [Tim Adams](https://www.researchgate.net/profile/Tim-Adams-3)

### Acknowledgments

This work was supported by the Fraunhofer Internal Programs under Grant No. 836 885.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
