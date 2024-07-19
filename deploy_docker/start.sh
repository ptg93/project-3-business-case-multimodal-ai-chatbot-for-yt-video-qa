#!/bin/sh

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

export FLASK_APP=app.py
flask run --host=0.0.0.0