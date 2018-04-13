

### Instructions

1. Install python requirements: `pip install -r minigui/requirements.txt`

1. Download [wgo.js](https://github.com/waltheri/wgo.js/blob/master/wgo/wgo.js) and put it in minigui/static

1. Download a model from our [public bucket](https://console.cloud.google.com/storage/browser/minigo-pub) and set the path appropriately in `serve.py` where it says `MODEL_PATH = ...`. 

1. Compile the typescript. (Requires [typescript compiler](https://www.typescriptlang.org/#download-links)). Running `cd minigui; tsc` should find and compile the relevant files.

1. Set your current working directory to minigo root and start the flask server.
```
FLASK_DEBUG=1 FLASK_APP=minigui/serve.py flask run --port 5001
```

1. open localhost:5001.

1. ...profit?
