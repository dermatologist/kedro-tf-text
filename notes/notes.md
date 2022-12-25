## Multiple outputs:
```
pipeline = [
    node(
        split_data,
        ["example_iris_data", "params:example_test_data_ratio"],
        dict(
            train_x="example_train_x",
            train_y="example_train_y",
            test_x="example_test_x",
            test_y="example_test_y",
        ),
    )
]

```

## Processed text as pkl

### To load
#### taking the intersection of ids of both datasets
ids   = list(set(list(text.keys())) & set(list(img.keys())))
text  = [text[patient] for patient in ids]
img   = [img[patient] for patient in ids]
y     = [original_data[original_data['ID'] == patient].Labels.item() for patient in ids]
## Ref: https://github.com/faikaydin/medical-multimodal-with-transfer-learning

    last_layer = deepmodel.model.layers[-2].output
    text_last_layer = BatchNormalization()(last_layer)
    return text_last_layer

## Pretrained embeddings

* https://github.com/gweissman/clinical_embeddings
* https://vaclavkosar.com/ml/Multimodal-Image-Text-Classification

## References
* https://github.com/artelab/Image-and-Text-fusion-for-UPMC-Food-101-using-BERT-and-CNNs  **Add next**
* https://github.com/AxelAllen/Multimodal-BERT-in-Medical-Image-and-Text-Classification
* https://github.com/faikaydin/medical-multimodal-with-transfer-learning
## [Kedro packaging](https://kedro.readthedocs.io/en/stable/tutorial/package_a_project.html)
* kedro package
* pip install <path-to-wheel-file>
* python -m package_name

## [Modular pipelines](https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/modular_pipelines.html)
* kedro pipeline create <pipeline_name>
* [from site]The kedro.pipeline.modular_pipeline.pipeline wrapper method unlocks the real power of modular pipelines. I don't take this approach. **I pass a dict to the constructor**
* Us this approach when using third party modular pipelines.
### [Micropackage](https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/micro_packaging.html)
* kedro micropkg package <micropkg_name> Example: kedro micropkg package pipelines.preprocess
* kedro micropkg package --alias <new_package_name> <micropkg_name>  Example: kedro micropkg package --alias kedro_tf_text pipelines.preprocess
* kedro micropkg pull           Example:kedro micropkg pull dist/preprocess-0.1.tar.gz

