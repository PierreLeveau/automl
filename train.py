import os

import click
from kili.client import Kili
from tabulate import tabulate

from utils.constants import ContentInput, HOME, InputType, MLTask, \
    ModelFramework, ModelName, ModelRepository, Tool
from utils.helpers import get_assets, get_project, kili_print, set_default

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'


def train_image_bounding_box(
        api_key, assets, job, job_name,
        model_framework, model_name, model_repository, project_id):
    '''
    '''
    from utils.ultralytics.train import ultralytics_train_yolov5
    model_repository = set_default(model_repository, ModelRepository.PyTorchHub,
        'model_repository', [ModelRepository.PyTorchHub])
    path = os.path.join(HOME, project_id, job_name, model_repository)
    if model_repository == ModelRepository.PyTorchHub:
        model_framework = set_default(model_framework, ModelFramework.PyTorch,
            'model_framework', [ModelFramework.PyTorch])
        model_name = set_default(model_name, ModelName.YoloV5,
            'model_name', [ModelName.YoloV5])
        return ultralytics_train_yolov5(
            api_key, assets, job, job_name, model_framework, model_name, path)


def train_ner(
        api_key, assets, job, job_name,
        model_framework, model_name, model_repository, project_id):
    '''
    '''
    from utils.huggingface.train import huggingface_train_ner
    import nltk
    nltk.download('punkt')
    model_repository = set_default(model_repository, ModelRepository.HuggingFace,
        'model_repository', [ModelRepository.HuggingFace])
    path = os.path.join(HOME, project_id, job_name, model_repository)
    if model_repository == ModelRepository.HuggingFace:
        model_framework = set_default(model_framework, ModelFramework.PyTorch,
            'model_framework', [ModelFramework.PyTorch, ModelFramework.Tensorflow])
        model_name = set_default(model_name, ModelName.BertBaseMultilingualCased,
            'model_name', [ModelName.BertBaseMultilingualCased])
        return huggingface_train_ner(
            api_key, assets, job, job_name, model_framework, model_name, path)


def train_text_classification_single(
        api_key, assets, job, job_name,
        model_framework, model_name, model_repository, project_id) -> float:
    '''
    '''
    import nltk
    nltk.download('punkt')
    from utils.huggingface.train import huggingface_train_text_classification_single
    model_repository = set_default(model_repository, ModelRepository.HuggingFace,
        'model_repository', [ModelRepository.HuggingFace])
    path = os.path.join(HOME, project_id, job_name, model_repository)
    if model_repository == ModelRepository.HuggingFace:
        model_framework = set_default(model_framework, ModelFramework.PyTorch,
            'model_framework', [ModelFramework.PyTorch, ModelFramework.Tensorflow])
        model_name = set_default(model_name, ModelName.BertBaseMultilingualCased,
            'model_name', [ModelName.BertBaseMultilingualCased])
        return huggingface_train_text_classification_single(
            api_key, assets, job, job_name, model_framework, model_name, path)




@click.command()
@click.option('--api-key', default=os.environ["KILI_API_KEY"], help='Kili API Key')
@click.option('--model-framework', default=None, help='Model framework (eg. pytorch, tensorflow)')
@click.option('--model-name', default=None, help='Model name (eg. bert-base-cased)')
@click.option('--model-repository', default=None, help='Model repository (eg. huggingface)')
@click.option('--project-id', default=None, help='Kili project ID')
@click.option('--label-types', default=None, help='Comma separated list Kili specific label types to select (among DEFAULT, REVIEW, PREDICTION)')
def main(api_key: str, model_framework: str, model_name: str, model_repository: str, project_id: str, label_types: str):
    '''
    '''
    kili = Kili(api_key=api_key)
    input_type, jobs = get_project(kili, project_id)
    assets = get_assets(kili, project_id, label_types.split(",") if label_types else None)
    training_losses = []
    for job_name, job in jobs.items():
        content_input = job.get('content', {}).get('input')
        ml_task = job.get('mlTask')
        tools = job.get('tools')
        training_loss = None
        print("Kili job properties")
        print(f"input: {content_input}")
        print(f"ml_task: {ml_task}")
        print(f"tools: {tools}")

        if content_input == ContentInput.Radio \
                and input_type == InputType.Text \
                and ml_task == MLTask.Classification:
            training_loss = train_text_classification_single(
                api_key, assets, job, job_name,
                model_framework, model_name, model_repository, project_id)
        elif content_input == ContentInput.Radio \
                and input_type == InputType.Text \
                and ml_task == MLTask.NamedEntitiesRecognition:
            training_loss = train_ner(
                api_key, assets, job, job_name,
                model_framework, model_name, model_repository, project_id)
        elif content_input == ContentInput.Radio \
                and input_type == InputType.Image \
                and ml_task == MLTask.ObjectDetection \
                and Tool.Rectangle in tools:
            training_loss = train_image_bounding_box(
                api_key, assets, job, job_name,
                model_framework, model_name, model_repository, project_id)
        else:
            kili_print('not implemented yet')
        training_losses.append([job_name, training_loss])
    kili_print()
    print(tabulate(training_losses, headers=['job_name', 'training_loss']))


if __name__ == '__main__':
    main()