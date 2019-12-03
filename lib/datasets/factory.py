"""
@Author: Ross Girshick
@Description: Factory method for easily getting imdbs by name.
"""

from datasets.psdb import psdb

__sets = {}

# Set up psdb_<split>
for split in ["train", "test"]:
    name = "psdb_{}".format(split)
    __sets[name] = lambda split=split: psdb(split)


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError("Unknown dataset: {}".format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
