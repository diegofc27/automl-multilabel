# End to end deep multilabel clasiffication

### Description
------------
End to end system for multilabel clasiffication using ResNet blocks and Differential Evolution Hyperband

### Usage
------------

To run the ResNet with DEHB and run:
```bash
$ python main.py --dataset 40589 --min_budget 25 --max_budget 500 --min 55
>Validation F1: 0.5034917192288638
>Final Test F1: 0.5660981368761254
```
List of avalible datasets
```bash
birds       0.132177
emotions    0.624649
enron       0.154276
genbase     0.716049
image       0.471796
langLog     0.007548
reuters     0.549591
scene       0.692282
slashdot    0.233592
yeast       0.326794
```