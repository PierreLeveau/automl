import os
from pyexpat import features
from typing import Dict, List, Optional
from datetime import datetime
import shutil

from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd

from utils.helpers import categories_from_job, kili_print

env = Environment(
    loader=FileSystemLoader(os.path.abspath("utils/ultralytics")), autoescape=select_autoescape()
)


def ultralytics_train_yolov5(
    api_key: str,
    path: str,
    job: Dict,
    max_assets: Optional[int],
    json_args: Dict,
    project_id: str,
    model_framework: str,
    label_types: List[str]
) -> float:
    yolov5_path = "utils/ultralytics/yolov5"

    template = env.get_template("kili_template.yml")
    class_names = categories_from_job(job)
    data_path = os.path.join(path, "data")
    config_data_path = os.path.join(yolov5_path, "data", "kili.yaml")
    output_path = os.path.join(path, 'model', model_framework,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs(output_path)

    with open(config_data_path, "w") as f:
        f.write(
            template.render(
                data_path=data_path,
                class_names=class_names,
                number_classes=len(class_names),
                kili_api_key=api_key,
                project_id=project_id,
                label_types=label_types,
                max_assets=max_assets
            )
        )

    args_from_json = " ".join(f" --{k} {v}" for k, v in json_args.items())
    kili_print("Starting Ultralytics\' YoloV5 ...")
    os.system(f'cd {yolov5_path} && python train.py --data kili.yaml --project "{output_path}" {args_from_json}')
    shutil.copy(config_data_path, output_path)
    df_result = pd.read_csv(os.path.join(output_path, "exp", "results.csv"))

    # we take the class loss as the main metric
    return df_result.iloc[-1:][['        val/obj_loss']].to_numpy()[0][0]
