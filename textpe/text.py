import os
from pe.data.text.text_csv import TextCSV

class text(TextCSV):
    """The text dataset specially implemented for PE algorithm"""

    def __init__(self, root_dir="data", **kwargs):
        """Constructor.

        :param root_dir: The root directory of the dataset. If the dataset is not there, it will be downloaded
            automatically. Defaults to "data"
        :type root_dir: str, optional
        """
        self._data_path = root_dir
        super().__init__(csv_path=self._data_path, label_columns=[], text_column="text", **kwargs)
