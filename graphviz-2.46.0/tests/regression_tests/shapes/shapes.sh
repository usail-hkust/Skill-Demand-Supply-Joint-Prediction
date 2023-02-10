#!/bin/sh -e

cd `dirname $0`

if command -v python3; then
    python3 shapes.py
else
    python2 shapes.py
fi
