#!/bin/bash
cd $PYBUILD/glue
$PYTHON setup.py build
$PYTHON setup.py sdist
$PYTHON setup.py install
