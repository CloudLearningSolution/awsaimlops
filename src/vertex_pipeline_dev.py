"""
Vertex AI KFP Pipeline for Development

This pipeline performs the following steps:
- Queries diabetes data from BigQuery Feature Group view (splits via query logic).
- Trains a logistic regression model in development mode.
- Evaluates the trained model on a test split.
- Conditionally registers the model in Vertex AI Model Registry if accuracy
  meets the minimum threshold.
- Rejects the model if accuracy is insufficient.

All comments and documentation lines are kept <= 100 characters for .flake8.

===============================================================================
Lab 5.4: Vertex AI Pipeline Component Architecture Exploration
===============================================================================
This pipeline demonstrates the complete Vertex AI Pipeline component architecture:
- KFP v2 component definitions using @component decorator
- Pipeline orchestration using @pipeline decorator
- Component dependencies and data flow
- Conditional logic using dsl.If

===============================================================================
Lab 5.5: Vertex AI Custom Components and Pre-built Components and Accelerator Templates
===============================================================================
This pipeline demonstrates the strategic use of custom components and how they
compare to pre-built Google Cloud Pipeline Components.

ACCELERATOR TEMPLATE ARCHITECTURE (3-Layer Enterprise Model):
=============================================================
1. INFRASTRUCTURE LAYER (Terraform)
2. PIPELINE LAYER (This File - Vertex AI)
3. ENTERPRISE LAYER (Best Practices)

COMPONENT STRATEGY IN THIS PIPELINE:
====================================
This pipeline uses a MIX of custom and pre-built components.

BIGQUERY AND FEATURE GROUP INTEGRATION:
=======================================
This pipeline demonstrates enterprise-grade BigQuery integration.
===============================================================================
Anushka was here!
"""

from kfp import dsl, components
from kfp.dsl import (
    component,
    pipeline,
    Input,
    Output,
    Model,
    Metrics
)
from google_cloud_pipeline_components.types import artifact_types

PIPELINE_NAME = "mlops-diabetes-dev-pipeline"
PIPELINE_DESCRIPTION = (
    "Development pipeline for diabetes prediction model on Vertex AI with BigQuery"
)

BASE_IMAGE = "python:3.9"
REQUIREMENTS_PATH = "src/requirements.txt"

bigquery_query_job_op = components.load_component_from_url(
    'https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/'
    'bigquery-query-job/sha256:'
    'd1cae80bc0de4e5b95b994739c8d0d7d42ce5a4cb17d3c9512eaed14540f6343'
)


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open(REQUIREMENTS_PATH)
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def train_model_op(
    train_data: Input[artifact_types.BQTable],
    output_model: Output[Model],
    reg_rate: float,
    project_id: str,
    bq_location: str
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
    logging.info("[DEV] Parsing BigQuery table from URI: %s", uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BigQuery table reference from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    logging.info("[DEV] Reading training data from BigQuery: %s", table_ref)
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    train_df = bq_client.query(query).to_dataframe()
    logging.info("[DEV] Loaded %d training rows from BigQuery", len(train_df))
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Diabetic"]

    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X, y)

    model_path = os.path.join(os.path.dirname(output_model.path), "model.joblib")
    joblib.dump(model, model_path)
    shutil.copy(model_path, output_model.path)
    logging.info(
        "[DEV] Model trained and stored at: %s and copied to: %s",
        model_path,
        output_model.path
    )


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open(REQUIREMENTS_PATH)
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def evaluate_model_op(
    test_data: Input[artifact_types.BQTable],
    model: Input[Model],
    metrics: Output[Metrics],
    min_accuracy: float,
    project_id: str,
    bq_location: str
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
    logging.info("[DEV] Parsing BigQuery table from URI: %s", uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BigQuery table reference from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    logging.info("[DEV] Reading test data from BigQuery: %s", table_ref)
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    test_df = bq_client.query(query).to_dataframe()
    logging.info("[DEV] Loaded %d test rows from BigQuery", len(test_df))
    model_artifact = joblib.load(model.path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    predictions = model_artifact.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("min_accuracy_threshold", min_accuracy)

    logging.info("[DEV] Accuracy = %.4f", accuracy)
    return accuracy


@component(
    base_image=BASE_IMAGE
)
def model_approved_op(model_accuracy: float, model: Input[Model]):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "[DEV] ✅ Model approved with accuracy: %.4f",
        model_accuracy
    )
    logging.info(
        "[DEV] Ready for registration from: %s",
        model.uri
    )


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open(REQUIREMENTS_PATH)
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def register_model_op(
    project_id: str,
    region: str,
    model_display_name: str,
    model_artifact: Input[Model],
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
            "[DEV] Registering new version under parent model: %s",
            parent_model
        )

    model = aiplatform.Model.upload(**upload_args)
    logging.info(
        "[DEV] Model registered: %s",
        model.resource_name
    )


@component(
    base_image=BASE_IMAGE
)
def model_rejected_op(model_accuracy: float, min_accuracy: float):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.error(
        "[DEV] ❌ Model rejected. Accuracy %.4f < %.2f",
        model_accuracy,
        min_accuracy
    )
    raise ValueError(
        "Model accuracy does not meet minimum development threshold."
    )


@dsl.pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def dev_diabetes_pipeline(
    project_id: str,
    region: str,
    model_display_name: str,
    bq_dataset: str,
    bq_view: str,
    reg_rate: float = 0.05,
    min_accuracy: float = 0.70,
    parent_model: str = ""
):
    train_query = f"""
    SELECT
      Pregnancies,
      PlasmaGlucose,
      DiastolicBloodPressure,
      TricepsThickness,
      SerumInsulin,
      BMI,
      DiabetesPedigree,
      Age,
      Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(entity_id AS STRING))), 10) < 8
    """
    bq_train_task = bigquery_query_job_op(
        project=project_id,
        location=region,
        query=train_query
    )

    test_query = f"""
    SELECT
      Pregnancies,
      PlasmaGlucose,
      DiastolicBloodPressure,
      TricepsThickness,
      SerumInsulin,
      BMI,
      DiabetesPedigree,
      Age,
      Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(entity_id AS STRING))), 10) >= 8
    """
    bq_test_task = bigquery_query_job_op(
        project=project_id,
        location=region,
        query=test_query
    )

    train_task = train_model_op(
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(bq_train_task)

    eval_task = evaluate_model_op(
        model=train_task.outputs["output_model"],
        test_data=bq_test_task.outputs["destination_table"],
        min_accuracy=min_accuracy,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    eval_task.after(train_task)

    with dsl.If(
        eval_task.outputs["Output"] >= min_accuracy,
        name="pass-accuracy-threshold"
    ):
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model=train_task.outputs["output_model"]
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        approved_task.after(eval_task)

        register_task = register_model_op(
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            model_artifact=train_task.outputs["output_model"],
            parent_model=parent_model
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        register_task.after(approved_task)

    with dsl.If(
        eval_task.outputs["Output"] < min_accuracy,
        name="fail-accuracy-threshold"
    ):
        rejected_task = model_rejected_op(
            model_accuracy=eval_task.outputs["Output"],
            min_accuracy=min_accuracy
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        rejected_task.after(eval_task)
