#! /usr/bin/env python
# coding: utf-8
#

"""Base Model"""


class BaseModel:
    """Base Model class of Atspy"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def forecast(self):
        raise NotImplementedError
