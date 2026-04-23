# Importing necessary libraries
import os
import glob
import matplotlib.pyplot as plt
import random
import cv2
from collections import Counter
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
