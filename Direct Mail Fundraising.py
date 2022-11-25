#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:37:22 2022

@author: judson
"""


#creating neural networks

#importing packages
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import dmba
from dmba import classificationSummary



fundraising_df = ()