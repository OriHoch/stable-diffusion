import os
import uuid
import json
import datetime
import traceback

from fastapi import FastAPI, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .txt2img import main


app = FastAPI()
security = HTTPBasic()


def get_outdir():
    return os.path.join(
        datetime.datetime.now().strftime("%Y/%m/%d/%H/%M/%S"),
        str(uuid.uuid1())
    )


SD_API_CONFIG_FILENAME = os.environ.get("SD_API_CONFIG_FILENAME", "/home/ubuntu/sd-api-config.json")

with open(SD_API_CONFIG_FILENAME, "r") as f:
    sd_api_config = json.load(f)


@app.get("/txt2img")
async def index(prompt: str, num_images: int = 1, skip_grid: bool = True,
                ddim_steps: int = 50, n_iter: int = 1, width: int = 512,
                height: int = 512,
                credentials: HTTPBasicCredentials = Depends(security),):
    try:
        assert f"{sd_api_config['username']}:{sd_api_config['password']}" == f"{credentials.username}:{credentials.password}"
        outdir = get_outdir()
        main({"prompt": prompt, "n_samples": num_images, "skip_grid": skip_grid,
              "ddim_steps": ddim_steps, "n_iter": n_iter, "W": width, "H": height,
              "outdir": os.path.join("/var/www/sd/output/", outdir)})
        return {"ok": True, "url": os.path.join(sd_api_config["sd_output_base_url"], outdir)}
    except:
        return {"ok": False, "traceback": traceback.format_exc()}
