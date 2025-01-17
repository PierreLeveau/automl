# pyright: reportPrivateImportUsage=false, reportOptionalCall=false
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from functools import reduce
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing_extensions import TypedDict

from kiliautoml.models._base_model import BaseModel
from kiliautoml.utils.constants import (
    HOME,
    MLTaskT,
    ModelFrameworkT,
    ModelNameT,
    ModelRepositoryT,
)
from kiliautoml.utils.download_assets import download_project_images
from kiliautoml.utils.helpers import (
    JobPredictions,
    categories_from_job,
    get_label,
    get_last_trained_model_path,
    kili_print,
)
from kiliautoml.utils.path import ModelPathT, Path, PathUltralytics
from kiliautoml.utils.type import AssetT, JobT, LabelMergeStrategyT

env = Environment(
    loader=FileSystemLoader(os.path.abspath(PathUltralytics.ULTRALYTICS_REL_PATH)),
    autoescape=select_autoescape(),
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AutoMLYoloException(Exception):
    pass


# TODO: Move to PathUltralytics
def get_id_from_path(path_yolov5_inference: str) -> str:
    return os.path.split(path_yolov5_inference)[-1].split(".")[0]


class CategoryNameConfidence(TypedDict):
    name: str
    # confidence is a probability between 0 and 100.
    confidence: int


class BBoxAnnotation(TypedDict):
    boundingPoly: Any
    categories: List[CategoryNameConfidence]
    type: str


def inspect(e):
    kili_print("Error while executing YoloV5:")
    for k, v in e.__dict__.items():
        print(k)
        if isinstance(v, bytes):
            print(v.decode("utf-8"))
        else:
            print(v)


class UltralyticsObjectDetectionModel(BaseModel):

    ml_task: MLTaskT = "OBJECT_DETECTION"
    model_repository: ModelRepositoryT = "ultralytics"

    def __init__(
        self,
        *,
        project_id: str,
        job: JobT,
        job_name: str,
        model_name: ModelNameT,
        model_framework: ModelFrameworkT,
    ):
        BaseModel.__init__(
            self,
            job=job,
            job_name=job_name,
            model_name=model_name,
            model_framework=model_framework,
        )
        self.project_id = project_id

    def train(
        self,
        *,
        assets: List[AssetT],
        label_merge_strategy: LabelMergeStrategyT,
        epochs: int,
        batch_size: int,
        clear_dataset_cache: bool,
        disable_wandb: bool,
        verbose: int,
        title: str,
        json_args: Dict,  # type: ignore
        api_key: str,
    ):
        _ = verbose

        model_repository_dir = Path.model_repository_dir(
            HOME, self.project_id, self.job_name, self.model_repository
        )

        yolov5_path = os.path.join(os.getcwd(), PathUltralytics.YOLOV5_REL_PATH)

        template = env.get_template("kili_template.yml")
        class_names = categories_from_job(self.job)
        data_path = os.path.join(model_repository_dir, "data")
        config_data_path = os.path.join(yolov5_path, "data", "kili.yaml")

        if clear_dataset_cache and os.path.exists(data_path) and os.path.isdir(data_path):
            kili_print("Dataset cache for this project is being cleared.")
            shutil.rmtree(data_path)

        model_output_path = self._get_output_path_bbox(
            title, model_repository_dir, self.model_framework
        )
        os.makedirs(model_output_path, exist_ok=True)

        os.makedirs(os.path.dirname(config_data_path), exist_ok=True)
        self._yaml_preparation(
            data_path=data_path,
            class_names=class_names,
            kili_api_key=api_key,
            assets=assets,
            job_name=self.job_name,
            label_merge_strategy=label_merge_strategy,
        )

        with open(config_data_path, "w") as f:
            f.write(
                template.render(
                    data_path=data_path,
                    class_names=class_names,
                    number_classes=len(class_names),
                )
            )

        if not json_args:
            json_args = {}
        json_args["epochs"] = epochs
        args_from_json = reduce(
            lambda x, y: x + y, ([f"--{k}", f"{v}"] for k, v in json_args.items())
        )
        kili_print("Starting Ultralytics' YoloV5 ...")
        try:
            yolo_env = os.environ.copy()
            if disable_wandb:
                yolo_env["WANDB_DISABLED"] = "false"
            args = [
                "python",
                "train.py",
                "--data",
                "kili.yaml",
                "--project",
                f"{model_output_path}",
                "--upload_dataset",  # wandb
                "--weights",
                "yolov5n.pt",
                "--batch",
                str(batch_size),
                *args_from_json,
            ]
            print("Executing Yolo with command line:", " ".join(args))

            with open("/tmp/test.log", "wb") as f:
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=yolov5_path,
                    env=yolo_env,
                )
                for line in iter(process.stdout.readline, b""):  # type:ignore
                    sys.stdout.write(line.decode(sys.stdout.encoding))

                print("process return code:", process.returncode)
                output, error = process.communicate()
                if process.returncode != 0:
                    print(output)
                    print(error)
                    raise AutoMLYoloException()
        except subprocess.CalledProcessError as e:
            inspect(e)

            raise AutoMLYoloException()

        shutil.copy(config_data_path, model_output_path)
        df_result = pd.read_csv(os.path.join(model_output_path, "exp", "results.csv"))

        # we take the class loss as the main metric
        return df_result.iloc[-1:][["        val/obj_loss"]].to_numpy()[0][0]  # type: ignore

    @staticmethod
    def _yaml_preparation(
        *,
        data_path: str,
        class_names: List[str],
        kili_api_key: str,
        assets,
        job_name,
        label_merge_strategy,
    ):

        print("Downloading datasets from Kili")
        train_val_proportions = [0.8, 0.1]
        path = data_path
        if "/kili/" not in path:
            raise ValueError("'path' field in config must contain '/kili/'")

        n_train_assets = math.floor(len(assets) * train_val_proportions[0])
        n_val_assets = math.floor(len(assets) * train_val_proportions[1])
        assert (
            n_val_assets > 0
        ), "Validation set must contain at least 2 assets. max_asset should be > 9"
        assets_splits = {
            "train": assets[:n_train_assets],
            "val": assets[n_train_assets : n_train_assets + n_val_assets],
            "test": assets[n_train_assets + n_val_assets :],
        }

        for name_split, assets_split in assets_splits.items():
            if len(assets_split) == 0:
                raise Exception("No asset in dataset, exiting...")
            path_split = os.path.join(path, "images", name_split)

            download_project_images(
                api_key=kili_api_key,
                assets=assets_split,
                output_folder=path_split,
            )

            names = class_names
            path_labels = re.sub("/images/", "/labels/", path_split)
            print(path_labels)
            os.makedirs(path_labels, exist_ok=True)
            for asset in assets_split:
                label = get_label(asset, label_merge_strategy)
                if (label is None) or (job_name not in label["jsonResponse"]):
                    asset_id = asset["id"]
                    warnings.warn(f"assetId {asset_id}: No annotation for job ${job_name}")
                else:
                    asset_id = asset["id"] + ".txt"  # type: ignore
                    with open(os.path.join(path_labels, asset_id), "w") as handler:
                        json_response = label["jsonResponse"]
                        for job in json_response.values():
                            save_annotations_to_yolo_format(names, handler, job)

    # TODO: Move to Paths
    @staticmethod
    def _get_output_path_bbox(title: str, path: str, model_framework: str) -> str:
        """Output folder path for the model."""
        # wandb splits the output_path according to "/" and uses the title as name of the project.
        model_output_path = os.path.join(
            path,
            "model",
            model_framework,
            datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            title,  # <-- name of the project in wandb
        )
        illegal_caracters = [",", "#", "?", "%", ":"]
        for char in illegal_caracters:
            model_output_path = model_output_path.replace(char, "_")
        kili_print("output_path of the model:", model_output_path)
        return model_output_path

    def predict(
        self,
        *,
        assets: List[AssetT],
        model_path: Optional[str],
        from_project: Optional[str],
        batch_size: int,
        verbose: int,
        clear_dataset_cache: bool,
        api_key: str = "",
    ):
        _ = clear_dataset_cache

        project_id = from_project if from_project else self.project_id

        model_path, model_framework = self._get_last_model_param(project_id, model_path)

        return self._predict(
            api_key,
            assets,
            project_id,
            model_framework,
            model_path,
            self.job_name,
            verbose,
            batch_size,
            prioritization=False,
        )

    def _predict(
        self,
        api_key: str,
        assets: List[AssetT],
        project_id: str,
        model_framework: ModelFrameworkT,
        model_path: ModelPathT,
        job_name: str,
        verbose: int,
        batch_size: int,
        prioritization: bool,
    ) -> JobPredictions:
        _ = batch_size

        warnings.warn("This function does not support custom batch_size")

        if model_framework == "pytorch":
            filename_weights = "best.pt"
        else:
            raise NotImplementedError(
                f"Predictions with model framework {model_framework} not implemented"
            )

        kili_print(f"Loading model {model_path}")
        kili_print(f"for job {job_name}")
        with open(os.path.join(model_path, "..", "..", "kili.yaml")) as f:
            kili_data_dict = yaml.load(f, Loader=yaml.FullLoader)

        inference_path = PathUltralytics.inference_dir(HOME, project_id, job_name, "ultralytics")
        model_weights = os.path.join(model_path, filename_weights)

        # path needs to be cleaned-up to avoid inferring unnecessary items.
        if os.path.exists(inference_path) and os.path.isdir(inference_path):
            shutil.rmtree(inference_path)
        os.makedirs(inference_path)

        downloaded_images = download_project_images(api_key, assets, output_folder=inference_path)
        # https://github.com/ultralytics/yolov5/blob/master/detect.py
        # default  --conf-thres=0.25, --iou-thres=0.45
        prioritizer_args = " --conf-thres=0.01  --iou-thres=0.45 " if prioritization else ""

        kili_print("Starting Ultralytics' YoloV5 inference...")
        cmd = (
            "python detect.py "
            + f'--weights "{model_weights}" '
            + "--save-txt --save-conf --nosave --exist-ok "
            + f'--source "{inference_path}" --project "{inference_path}" '
            + prioritizer_args
        )
        os.system(
            "cd " + PathUltralytics.YOLOV5_REL_PATH + " && " + cmd
        )  # TODO: Use instead subcommand and catch errors

        inference_files = glob(os.path.join(inference_path, "exp", "labels", "*.txt"))
        inference_files_by_id = {get_id_from_path(pf): pf for pf in inference_files}

        kili_print("Converting Ultralytics' YoloV5 inference to Kili JSON format...")
        id_json_list: List[Tuple[str, Dict]] = []  # type: ignore

        proba_list: List[float] = []
        for image in downloaded_images:
            if image.id in inference_files_by_id:
                kili_predictions, probabilities = yolov5_to_kili_json(
                    inference_files_by_id[image.id], kili_data_dict["names"]
                )
                proba_list.append(min(probabilities))
                if verbose >= 1:
                    print(f"Asset {image.externalId}: {kili_predictions}")
                id_json_list.append(
                    (image.externalId, {job_name: {"annotations": kili_predictions}})
                )

        # TODO: move this check in the prioritizer
        if len(id_json_list) < len(downloaded_images):
            kili_print(
                "WARNING: Not enouth predictions. Missing prediction for"
                f" {len(downloaded_images) - len(id_json_list)} assets."
            )
            if prioritization:
                # TODO: Automatically tune the threshold
                # TODO: Do not crash and put 0 probability for missing assets.
                raise Exception(
                    "Not enough predictions for prioritization. You should either train longer the"
                    " model, or lower the --conf-thres "
                )

        job_predictions = JobPredictions(
            job_name=job_name,
            external_id_array=[a[0] for a in id_json_list],
            json_response_array=[a[1] for a in id_json_list],
            model_name_array=["Kili AutoML"] * len(id_json_list),
            predictions_probability=proba_list,
        )
        print("predictions_probability", job_predictions.predictions_probability)
        return job_predictions

    def _get_last_model_param(self, project_id, model_path) -> Tuple[ModelPathT, ModelFrameworkT]:
        model_path = get_last_trained_model_path(
            project_id=project_id,
            job_name=self.job_name,
            project_path_wildcard=[
                "ultralytics",  # ultralytics or huggingface # TODO: only ultralytics
                "model",
                "*",  # pytorch or tensorflow
                "*",  # date and time
                "*",  # title of the project, but already specified by project_id
                "exp",
                "weights",
            ],
            weights_filename="best.pt",
            model_path=model_path,
        )

        split_path = os.path.normpath(model_path).split(os.path.sep)  # type: ignore

        model_framework: ModelFrameworkT = split_path[-5]  # type: ignore
        kili_print(f"Model framework: {model_framework}")
        if model_framework not in ["pytorch", "tensorflow"]:
            raise ValueError(f"Unknown model framework: {model_framework}")
        return model_path, model_framework

    def find_errors(
        self,
        *,
        assets: List[AssetT],
        label_merge_strategy: LabelMergeStrategyT,
        cv_n_folds: int,
        epochs: int,
        batch_size: int,
        verbose: int = 0,
        clear_dataset_cache: bool = False,
        api_key: str = "",
    ) -> Any:
        pass


def save_annotations_to_yolo_format(names, handler, job):
    for annotation in job.get("annotations", []):
        name = annotation["categories"][0]["name"]
        try:
            category = names.index(name)
        except ValueError:
            pass
        bounding_poly = annotation.get("boundingPoly", [])
        if len(bounding_poly) < 1:
            continue
        if "normalizedVertices" not in bounding_poly[0]:
            continue
        normalized_vertices = bounding_poly[0]["normalizedVertices"]
        x_s = [vertice["x"] for vertice in normalized_vertices]
        y_s = [vertice["y"] for vertice in normalized_vertices]
        x_min, y_min = min(x_s), min(y_s)
        x_max, y_max = max(x_s), max(y_s)
        _x_, _y_ = (x_max + x_min) / 2, (y_max + y_min) / 2
        _w_, _h_ = x_max - x_min, y_max - y_min
        handler.write(f"{category} {_x_} {_y_} {_w_} {_h_}\n")  # type: ignore


def yolov5_to_kili_json(
    path_yolov5_inference: str, ind_to_categories: List[str]
) -> Tuple[List[BBoxAnnotation], List[int]]:
    """Returns a list of annotations and of probabilities"""
    annotations = []
    probabilities = []
    with open(path_yolov5_inference, "r") as f:
        for line in f.readlines():
            c_, x_, y_, w_, h_, p_ = line.split(" ")
            x, y, w, h = float(x_), float(y_), float(w_), float(h_)
            c = int(c_)
            p = int(100.0 * float(p_))

            category: CategoryNameConfidence = {
                "name": ind_to_categories[c],
                "confidence": p,
            }
            probabilities.append(p)

            bbox_annotation: BBoxAnnotation = {
                "boundingPoly": [
                    {
                        "normalizedVertices": [
                            {"x": x - w / 2, "y": y + h / 2},
                            {"x": x - w / 2, "y": y - h / 2},
                            {"x": x + w / 2, "y": y - h / 2},
                            {"x": x + w / 2, "y": y + h / 2},
                        ]
                    }
                ],
                "categories": [category],
                "type": "rectangle",
            }

            annotations.append(bbox_annotation)

    return (annotations, probabilities)
