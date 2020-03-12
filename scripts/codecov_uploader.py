from os import environ
from time import sleep
from codecov import main as codecov

for _ in range(10):
    try:
        print("Sending coverage information")
        codecov("--token", environ["CODECOV_TOKEN"], "--name", environ["NAME"], "--required")
    except Exception as exc:
        print("Codecov failed: {}".format(exc), flush=True)
        sleep(30)
        continue
    else:
        print("Codecov upload was successful")
        break
