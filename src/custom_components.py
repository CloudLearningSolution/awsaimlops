"""
Custom Kubeflow Pipelines (KFP) components for the Vertex AI MLOps workflow.

This module contains the definitions for all custom, project-specific pipeline
steps, such as model training, evaluation, and registration. By centralizing
these components, we ensure consistency and maintainability across different
pipelines (e.g., development and production).

All comments and documentation lines are kept <= 100 characters for .flake8.
"""

import os
import shutil
from kfp import dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Model,
    Metrics
)
from google_cloud_pipeline_components.types import artifact_types

# ------------------------------------------------------------------------------
# Component: train_model_op
# ------------------------------------------------------------------------------


@component(
    base_image="python:3.9",
    packages_to_install=[
        pkg.strip()
        for pkg in open("src/requirements.txt")
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def train_model_op(
    train_data: Input[artifact_types.BQTable],
    output_model: Output[Model],
    reg_rate: float,
    project_id: str,
    bq_location: str,
    env_prefix: str,
):
    import pandas as pd
    import joblib
    from sklearn.linear_model import LogisticRegression
    from google.cloud import bigquery
    import logging
    import os
    import shutil
    import re

    FEATURE_COLUMNS = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]

    logging.basicConfig(level=logging.INFO)
    uri = train_data.uri
    logging.info("%s Parsing BigQuery table from URI: %s", env_prefix, uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table reference from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    logging.info("%s Reading training data from BigQuery: %s", env_prefix, table_ref)
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    train_df = bq_client.query(query).to_dataframe()
    logging.info("%s Loaded %d training rows from BigQuery", env_prefix, len(train_df))
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Diabetic"]

    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X, y)

    model_path = os.path.join(os.path.dirname(output_model.path), "model.joblib")
    joblib.dump(model, model_path)
    shutil.copy(model_path, output_model.path)
    logging.info(
        "%s Model trained and stored at: %s and copied to: %s",
        env_prefix,
        model_path,
        output_model.path
    )

# ------------------------------------------------------------------------------
# Component: evaluate_model_op
# ------------------------------------------------------------------------------


@component(
    base_image="python:3.9",
    packages_to_install=[
        pkg.strip()
        for pkg in open("src/requirements.txt")
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def evaluate_model_op(
    test_data: Input[artifact_types.BQTable],
    model: Input[Model],
    metrics: Output[Metrics],
    min_accuracy: float,
    project_id: str,
    bq_location: str,
    env_prefix: str,
) -> float:
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score
    from google.cloud import bigquery
    import logging
    import re

    FEATURE_COLUMNS = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]

    logging.basicConfig(level=logging.INFO)
    uri = test_data.uri
    logging.info("%s Parsing BigQuery table from URI: %s", env_prefix, uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table reference from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    logging.info("%s Reading test data from BigQuery: %s", env_prefix, table_ref)
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    test_df = bq_client.query(query).to_dataframe()
    logging.info("%s Loaded %d test rows from BigQuery", env_prefix, len(test_df))
    model_artifact = joblib.load(model.path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    predictions = model_artifact.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("min_accuracy_threshold", min_accuracy)

    logging.info("%s Accuracy = %.4f", env_prefix, accuracy)
    return accuracy

# ------------------------------------------------------------------------------
# Component: model_approved_op
# ------------------------------------------------------------------------------


@component(
    base_image="python:3.9"
)
def model_approved_op(
    model_accuracy: float,
    model: Input[Model],
    env_prefix: str
):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "%s ✅ Model approved with accuracy: %.4f",
        env_prefix,
        model_accuracy
    )
    logging.info(
        "%s Ready for registration from: %s",
        env_prefix,
        model.uri
    )

# ------------------------------------------------------------------------------
# Component: register_model_op
# ------------------------------------------------------------------------------


@component(
    base_image="python:3.9",
    packages_to_install=[
        pkg.strip()
        for pkg in open("src/requirements.txt")
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def register_model_op(
    project_id: str,
    region: str,
    model_display_name: str,
    model_artifact: Input[Model],
    env_prefix: str,
    parent_model: str = ""
):
    from google.cloud import aiplatform
    import logging

    logging.basicConfig(level=logging.INFO)
    aiplatform.init(project=project_id, location=region)

    artifact_dir = model_artifact.uri.rsplit("/", 1)[0]
    upload_args = {
        "display_name": model_display_name,
        "artifact_uri": artifact_dir,
        "serving_container_image_uri": (
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"
        ),
        "sync": True
    }

    if parent_model:
        upload_args["parent_model"] = parent_model
        logging.info(
            "%s Registering new version under parent model: %s",
            env_prefix,
            parent_model
        )

    model = aiplatform.Model.upload(**upload_args)
    logging.info(
        "%s Model registered: %s",
        env_prefix,
        model.resource_name
    )

# ------------------------------------------------------------------------------
# Component: model_rejected_op
# ------------------------------------------------------------------------------


@component(
    base_image="python:3.9"
)
def model_rejected_op(
    model_accuracy: float,
    min_accuracy: float,
    env_prefix: str,
):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.error(
        "%s ❌ Model rejected. Accuracy %.4f < %.2f",
        env_prefix,
        model_accuracy,
        min_accuracy
    )
    raise ValueError(
        "Model accuracy does not meet minimum threshold."
    )