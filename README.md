# AutoML multilabel
Refer to the [pdf](./project_automl_multilabel.pdf) for details on the assignment.

We have handled the minimum of data retrieval and basic prepreocessing to get the
baseline `RandomForest` to run and all of this is contained in [`data.py`](./data.py).

To run the baseline, ensure you are in this directory and run:
```bash
$ python baseline.py
>
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

In here you can see how to load in the data and pass it to a model, the rest is up to you!
# automl-multilabel
