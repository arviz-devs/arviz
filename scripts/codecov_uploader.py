from os import environ
from time import sleep
from codecov import main as codecov

for _ in range(10):
    try:
        codecov(["--token", environ["CODECOV_TOKEN"], "--name", environ["NAME"], "--required"])
    except Exception:
        sleep(30)
        continue
    else:
        break
