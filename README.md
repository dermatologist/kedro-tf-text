# Kedro Tf Text :scroll:

This package consists of [Kedro pipelines](https://kedro.readthedocs.io/en/stable/kedro.pipeline.html) for preprocessing text and tabular data for multimodal machine learning in healthcare. This package preprocesses BERT and CNN text models. Use [kedro-tf-image](https://github.com/dermatologist/kedro-tf-image) for preprocessing image models. The [kedro-tf-utils](https://github.com/dermatologist/kedro-tf-utils) creates **fusion** models for **training**. The [kedro-multimodal](https://github.com/dermatologist/kedro-multimodal) template uses these pipelines. **End users can just fork [kedro-multimodal](https://github.com/dermatologist/kedro-multimodal) template to build multimodal pipelines!**


[![kedro-tf-text](https://github.com/dermatologist/kedro-tf-text/blob/develop/notes/text.drawio.svg)](https://github.com/dermatologist/kedro-tf-text/blob/develop/notes/text.drawio.svg)

## How to install
```

pip install git+https://github.com/dermatologist/kedro-tf-text.git

```
## Pipelines
| Name | Input | Output | Description | Params |
| ---- | ---- | ---- | ---- | ---- |
| bert.download_bert | ["bert_model", "params:bert_model"] | "bert_model_saved" | Download and save bert model (See bert_model and bert_model_saved in catalog) | None |
| cnn.cnn_text_pipeline | ["glove_embedding", "params:cnn_text_model"] | cnn_text_model | creates a CNN text model from GloVe embedding layer | MAX_SEQ_LENGTH |
| preprocess.glove_embedding | ["text_data", "params:embedding"] | "glove_embedding" (Pickle) | Create GloVe embedding | REPORT_FIELD, ID, TARGET |
| preprocess.process_text_pipeline | ["text_data", "word2vec_embedding", "params:embedding"] | "processed_text" (Pickle) | process text using the word index from Word2Vec model | REPORT_FIELD, ID, TARGET |
| tabular.tabular_model_pipeline | ["tabular_data", "params:tabular"] | tabular_model (Pickle) | Create a model from tabular csv data | DROP, TARGET, EPOCHS, DENSE_LAYER |

## Catalog
```
bert_model:
  type: kedro_tf_text.extras.datasets.bert_model_download.BertModelDownload
  preprocessor_url: "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
  encoder_url: "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

bert_model_saved:
  type: tensorflow.TensorFlowModelDataset
  filepath: data/06_models/bert-tf

## ID | Long line of text | outcome (y)
text_data:
  type: pandas.CSVDataSet
  filepath: data/01_raw/test_report.csv

## ID | included | fields | excluded | fields | outcome (y)
tabular_data:
  type: pandas.CSVDataSet
  filepath: data/01_raw/test_dataset.csv

word2vec_embedding:
  type: pickle.PickleDataSet
  filepath: data/06_models/word2vec-embedding.pkl

glove_embedding:
  type: pickle.PickleDataSet
  filepath: data/06_models/glove-embedding.pkl

tabular_model:
  type: pickle.PickleDataSet
  filepath: data/06_models/tabular_model.pkl

processed_text:
  type: pickle.PickleDataSet
  filepath: data/03_primary/processed-text.pkl

fusion_model:
  type: tensorflow.TensorFlowModelDataset
  filepath: data/07_model_output/fusion

datasetinmemory:
  type: MemoryDataSet
  copy_mode: assign

```

## Author

- [Bell Eapen](https://nuchange.ca) [![Twitter Follow](https://img.shields.io/twitter/follow/beapen?style=social)](https://twitter.com/beapen)

## Overview

This is your new Kedro project, which was generated using `Kedro 0.18.1`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://kedro.readthedocs.io/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.

## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will `pip-compile` the contents of `src/requirements.txt` into a new file `src/requirements.lock`. You can see the output of the resolution by opening `src/requirements.lock`.

After this, if you'd like to update your project requirements, please update `src/requirements.txt` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r src/requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to convert notebook cells to nodes in a Kedro project
You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#release-5-0-0) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://kedro.readthedocs.io/en/stable/tutorial/package_a_project.html)
