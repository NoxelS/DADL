#!/usr/bin/env bash

# Check if conda is installed else just install the requirements.txt
if ! command -v conda &>/dev/null; then
    if ! command -v pip &>/dev/null; then
        pip install -r requirements.txt
    else
        if ! command -v pip3 &>/dev/null; then
            pip3 install -r requirements.txt
        else
        fi
    fi
else
    conda env export --name CNN --from-history --file environment.yml
fi
