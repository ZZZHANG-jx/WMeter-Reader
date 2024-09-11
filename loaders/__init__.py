import json

from loaders.text_text_loader_im_embedding_lmdb import TexttextLoader_lmdb
from loaders.text_text_loader_im_embedding_lmdb_create import TexttextLoader_lmdb_create

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'lmdb_create':TexttextLoader_lmdb_create,
        'lmdb':TexttextLoader_lmdb,
    }[name]
