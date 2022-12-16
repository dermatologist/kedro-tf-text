    last_layer = deepmodel.model.layers[-2].output
    text_last_layer = BatchNormalization()(last_layer)
    return text_last_layer

## [Kedro packaging](https://kedro.readthedocs.io/en/stable/tutorial/package_a_project.html)
* kedro package
* pip install <path-to-wheel-file>
* python -m package_name

## [Modular pipelines](https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/modular_pipelines.html)
* kedro pipeline create <pipeline_name>
* [from site]The kedro.pipeline.modular_pipeline.pipeline wrapper method unlocks the real power of modular pipelines. I don't take this approach. **I pass a dict to the constructor**
* Us this approach when using third party modular pipelines.
### [Micropackage](https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/micro_packaging.html)
* kedro micropkg package <micropkg_name>
* kedro micropkg pull

