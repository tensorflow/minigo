import google.cloud.logging as glog
import logging
import contextlib
import io

def configure():
    logging.basicConfig(level=logging.INFO)
    try:
        # if this fails, redirect stderr to /dev/null so no startup spam.
        with contextlib.redirect_stderr(io.StringIO()):
            client = glog.Client('tensor-go')
            client.setup_logging(logging.INFO)
    except:
        print('!! Cloud logging disabled')
