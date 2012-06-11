#!/bin/bash

type pip &> /dev/null

if [[ $? -eq 0 ]]
then
    COMMAND='pip'
fi

type pip2 &> /dev/null
if [[ $? -eq 0 ]]
then
    COMMAND='pip2'
fi

if [[ -z $COMMAND ]]
then
    echo "PyPi not found"
    exit 1
fi

PIP_PYTHON_VER=`$COMMAND --version | sed 's/.*python \([0-9]\.[0-9]\).*/\1/g'`

if [[ $PIP_PYTHON_VER != '2.7' ]]
then
    echo "Wrong Python version (must be 2.7, was $PIP_PYTHON_VER)"
    exit 1
fi

#Install required python packages
for PKG in numpy scipy matplotlib spectrum networkx ffnet
do
    echo "sudo $COMMAND install $PKG"
done
