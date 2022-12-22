from typing import Any, Dict, Optional
from kedro.io.core import AbstractDataSet
import tensorflow_hub as hub
import bert
from bert import tokenization

# https://gist.github.com/dermatologist/062c46eafe8c118334a004f6cfab663d
class BertModelDownload(AbstractDataSet):
    """This class downloads a BERT model and returns tokenizers and
    """

    def __init__(
        self,
        preprocessor_url: str,
        encoder_url: str,
    ) -> None:
        """Initialises the class.
        Args:
            filepath: The path to the file where the BERT model is saved.
            url: The URL from which the BERT model is downloaded.
            credentials: Credentials required to access the URL.
            version: If specified, should be an instance of
                ``kedro.io.core.Version``. If its ``load`` attribute is
                None, the latest version will be loaded. If its ``save``
                attribute is None, save version will be autogenerated.
        """
        self._preprocessor_url = preprocessor_url
        self._encoder_url = encoder_url

    def _load(self) -> Any:
        """Loads the BERT model from the URL and saves it to the specified
        location.
        """
        # TODO: https://github.com/tensorflow/hub/issues/705
        import tensorflow_text as text

        preprocessor = hub.KerasLayer(
            self._preprocessor_url,)
        encoder = hub.KerasLayer(
            self._encoder_url,
            trainable=True)
        return (preprocessor, encoder)

    def _save(self, data: Any) -> None:
        """Saves the BERT model to the specified location.
        Args:
            Not implemented
        """
        pass

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset.
        Returns:
            A dict that describes the attributes of the dataset.
        """
        return dict(
            preprocessor_url=self._preprocessor_url,
            encoder_url=self._encoder_url
            )

