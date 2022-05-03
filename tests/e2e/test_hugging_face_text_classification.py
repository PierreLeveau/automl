import json

from click.testing import CliRunner

import predict
import train

text_content = json.load(open("tests/e2e/fixtures/text_content_fixture.json"))


def mocked__get_text_from(asset_url):
    return text_content[asset_url]


def mocked__get_assets(*_, max_assets=None, labeling_statuses=None):
    return json.load(open("tests/e2e/fixtures/text_assets_fixture.json"))[:max_assets]


def mocked__projects(*_, project_id, fields):
    return json.load(open("tests/e2e/fixtures/text_project_fixture.json"))


def test_hugging_face_text_classification(mocker):

    mocker.patch("kili.client.Kili.__init__", return_value=None)
    mocker.patch("train.get_assets", side_effect=mocked__get_assets)
    mocker.patch("kili.client.Kili.projects", side_effect=mocked__projects)
    mocker.patch(
        "kiliautoml.mixins._kili_text_project_mixin.KiliTextProjectMixin._get_text_from",
        side_effect=mocked__get_text_from,
    )
    mock_create_predictions = mocker.patch("kili.client.Kili.create_predictions")

    runner = CliRunner()
    result = runner.invoke(
        train.main,
        [
            "--project-id",
            "abcdefgh",
            "--max-assets",
            "20",
            "--target-job",
            "CLASSIFICATION_JOB_0",
            "--model-name",
            "distilbert-base-cased",
            "--disable-wandb",
        ],
    )
    assert result.exception is None

    mocker.patch("predict.get_assets", side_effect=mocked__get_assets)
    result = runner.invoke(
        predict.main,
        [
            "--project-id",
            "abcdefgh",
            "--max-assets",
            "10",
            "--target-job",
            "CLASSIFICATION_JOB_0",
        ],
    )
    assert result.exception is None
    assert result.output.count("OPTIMISM") == 0
    mock_create_predictions.assert_called_once()
    # Note: useful for debugging:
    # import traceback
    # print(traceback.print_tb(result.exception.__traceback__))

    mock_create_predictions = mocker.patch("kili.client.Kili.create_predictions")
    result = runner.invoke(
        predict.main,
        [
            "--project-id",
            "abcdefgh",
            "--max-assets",
            "10",
            "--target-job",
            "CLASSIFICATION_JOB_0",
            "--dry-run",
            "--verbose",
            "1",
        ],
    )
    assert result.exception is None
    assert result.output.count("OPTIMISM") == 10
    mock_create_predictions.assert_not_called()