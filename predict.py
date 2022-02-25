from glob import glob
import os

import click
from kili.client import Kili

from utils.constants import (
    ContentInput,
    HOME,
    InputType,
    MLTask,
    ModelFramework,
    ModelRepository,
)
from utils.helpers import get_assets, get_project, kili_print


def predict_ner(api_key, assets, job, job_name, project_id):
    path_project_models = os.path.join(
        HOME, project_id, job_name, "*", "model", "*", "*"
    )
    paths_project_sorted = sorted(glob(path_project_models), reverse=True)
    model_path = None
    while len(paths_project_sorted):
        path_model_candidate = paths_project_sorted.pop(0)
        if len(os.listdir(path_model_candidate)) > 0 and os.path.exists(
            os.path.join(path_model_candidate, "pytorch_model.bin")
        ):
            model_path = path_model_candidate
            kili_print(f"Trained model found in path: {model_path}")
            break
    if model_path is None:
        raise Exception("No trained model found for job {job}. Exiting ...")
    split_path = os.path.normpath(model_path).split(os.path.sep)
    if split_path[-4] == ModelRepository.HuggingFace:
        model_repository = ModelRepository.HuggingFace
        kili_print(f"Model base repository: {model_repository}")
    else:
        raise ValueError("Unknown model base repository")
    if split_path[-2] in [ModelFramework.PyTorch, ModelFramework.Tensorflow]:
        model_framework = split_path[-2]
        kili_print(f"Model framework: {model_framework}")
    else:
        raise ValueError("Unknown model framework")
    if model_repository == ModelRepository.HuggingFace:
        from utils.huggingface.predict import huggingface_predict_ner

        return huggingface_predict_ner(api_key, assets, model_framework, model_path)


@click.command()
@click.option("--api-key", default=os.environ["KILI_API_KEY"], help="Kili API Key")
@click.option("--project-id", required=True, help="Kili project ID")
@click.option(
    "--label-types",
    default=None,
    help="Comma separated list Kili specific label types to select (among DEFAULT, REVIEW, PREDICTION)",
)
@click.option(
    "--dry-run",
    default=None,
    is_flag=True,
    help="Runs the predictions but do not save them into the Kili project",
)
def main(api_key: str, project_id: str, label_types: str, dry_run: bool):

    kili = Kili(api_key=api_key)
    input_type, jobs = get_project(kili, project_id)
    assets = get_assets(
        kili, project_id, label_types.split(",") if label_types else None
    )

    for job_name, job in jobs.items():
        content_input = job.get("content", {}).get("input")
        ml_task = job.get("mlTask")

        if (
            content_input == ContentInput.Radio
            and input_type == InputType.Text
            and ml_task == MLTask.NamedEntitiesRecognition
        ):
            outputs = predict_ner(api_key, assets, job, job_name, project_id)
        else:
            kili_print("not implemented yet")


if __name__ == "__main__":
    main()
