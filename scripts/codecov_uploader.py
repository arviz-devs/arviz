from subprocess import run, PIPE
from time import sleep

for _ in range(10):
    try:
        print("Sending coverage information")
        cmd = ["python", "-m", "codecov", "--token=$(CODECOV_TOKEN)", "--name=$(NAME)", "--required"]
        output = run(cmd, stdout=PIPE, stderr=PIPE)
        if "Error:" in output.stdout.decode("utf-8"):
            print("Error found in stdout")
            raise ValueError("Upload failed")
        elif "Error:" in output.stderr.decode("utf-8"):
            print("Error found in stderr")
            raise ValueError("Upload failed")
    except Exception as exc:
        print("Codecov upload failed: {}".format(exc), flush=True)
        sleep(30)
        continue
    else:
        print("Codecov upload was successful")
        break
