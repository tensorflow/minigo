

### Instructions

1. Install python requirements: `pip install -r minigui/requirements.txt`

1. Download a model from our [public bucket](https://console.cloud.google.com/storage/browser/minigo-pub) and set the path appropriately in `serve.py` where it says `MODEL_PATH = ...`. 

1. Make sure the command at the top of `serve.py` actually runs and prints `GTP engine ready`; if not, something is wrong with the rest of the minigo setup, like virtualenv or similar.

1. Compile the typescript. (Requires [typescript compiler](https://www.typescriptlang.org/#download-links)). Running `cd minigui; tsc` should find and compile the relevant files.

1. Set your current working directory to minigo root and start the flask server.
```
FLASK_DEBUG=1 FLASK_APP=minigui/serve.py flask run --port 5001
```

1. open localhost:5001.

1. The buttons in the upper right that say 'Human' can be toggled to set which color Minigo will play.
