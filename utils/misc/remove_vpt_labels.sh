#!/bin/bash

find vpt/data/numpy -type f -name "labels.txt" -exec rm -f {} \;
find vpt/data/numpy -type f -name "class_distribution_histogram.png" -exec rm -f {} \;
