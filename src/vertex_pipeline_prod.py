"""
Vertex AI KFP Pipeline for Production

This pipeline orchestrates the production workflow for the diabetes prediction model.
It includes the following steps:
- Queries diabetes data from BigQuery Feature Group view (splits via query logic).
- Trains a logistic regression model using scikit-learn.
- Evaluates the model on the test set and logs accuracy.
- Conditionally registers a new version of the model in Vertex AI Model Registry
  if accuracy meets the minimum threshold, supporting parent model versioning.
- Rejects the model if accuracy is below the required threshold.

All comments and documentation lines are kept <= 100 characters for .flake8.

===============================================================================
Lab 5.4: Vertex AI Pipeline Component Architecture Exploration
===============================================================================
This production pipeline demonstrates the same component architecture as the
development pipeline, but with production-specific configurations:
- Higher accuracy thresholds for quality control
- Production data sources and logging
- Model versioning support for production deployments
- Enhanced audit trail and monitoring

TODO: Lab 5.4.1 - Component Identification: Production pipeline with same component types as dev
TODO: Lab 5.4.2 - Purpose Recognition: Production-optimized configuration of components
TODO: Lab 5.4.3 - Architecture Understanding: Environment-specific pipeline architecture patterns

===============================================================================
Lab 5.5: Vertex AI Custom Components and Pre-built Components and Accelerator Templates
===============================================================================
This PRODUCTION pipeline demonstrates the same custom component strategy as the
development pipeline, with production-grade configurations and stricter quality gates.

ACCELERATOR TEMPLATE ARCHITECTURE (Production Environment):
==========================================================
1. INFRASTRUCTURE LAYER (Terraform):
   - Production Vertex AI resources with high availability
   - Separate service accounts for production isolation
   - Production GCS buckets with retention policies
   - Production BigQuery datasets and Feature Groups
   - Production-specific IAM and security controls
   - See: vertex_ai_infrastructure.tf (production workspace/environment)

2. PIPELINE LAYER (This File - Vertex AI Production):
   - Same components as dev (custom and pre-built mix)
   - Production configuration and parameters
   - Higher accuracy thresholds (0.75 vs 0.70)
   - Production data sources
   - Enhanced logging and audit trail
   - Model versioning for production lineage

3. ENTERPRISE LAYER (Production Best Practices):
   - Compliance and governance requirements
   - Production deployment workflows
   - Monitoring and alerting integration
   - Disaster recovery and backup strategies

COMPONENT STRATEGY - PRODUCTION CONSIDERATIONS:
==============================================
Production pipelines use the SAME COMPONENTS as development, but with:
- Stricter quality gates and thresholds
- Enhanced error handling and logging
- Production data sources and paths
- Model versioning support
- Integration with monitoring systems

TODO: Lab 5.5.1 - Custom vs Pre-built: Same components as dev, production configuration
TODO: Lab 5.5.2 - Pre-built Alternatives: Same BigQuery integration as dev
TODO: Lab 5.5.3 - Accelerator Templates: Production-grade 3-layer architecture
TODO: Lab 5.5.4 - Environment Reusability: Same components, different configurations
TODO: Lab 5.5.5 - BigQuery Integration: Production uses same data architecture as dev
===============================================================================
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

PIPELINE_NAME = "mlops-diabetes-prod-pipeline"
PIPELINE_DESCRIPTION = (
    "Production pipeline for diabetes prediction model on Vertex AI with BigQuery"
)

BASE_IMAGE = "python:3.9"
REQUIREMENTS_PATH = "src/requirements.txt"

# PRE-BUILT COMPONENT: BigQuery Query Job
bigquery_query_job_op = components.load_component_from_url(
    'https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/'
    'bigquery-query-job/sha256:'
    'd1cae80bc0de4e5b95b994739c8d0d7d42ce5a4cb17d3c9512eaed14540f6343'
)

# Component: train_model_op (MODIFIED FOR BIGQUERY - PRODUCTION)


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open(REQUIREMENTS_PATH)
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def train_model_op(
    train_data: Input[dsl.types.BQTable],  # <-- CHANGED
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

    FEATURE_COLUMNS = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]

    logging.basicConfig(level=logging.INFO)
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    table_uri = train_data.metadata.get("destinationTable", train_data.uri)
    logging.info("[PROD] Reading training data from BigQuery: %s", table_uri)
    query = f"SELECT * FROM `{table_uri}`"
    train_df = bq_client.query(query).to_dataframe()
    logging.info("[PROD] Loaded %d training rows from BigQuery", len(train_df))
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Diabetic"]

    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X, y)

    model_path = os.path.join(os.path.dirname(output_model.path), "model.joblib")
    joblib.dump(model, model_path)
    shutil.copy(model_path, output_model.path)
    logging.info(
        "[PROD] Model trained and stored at: %s and copied to: %s",
        model_path,
        output_model.path
    )

# Component: evaluate_model_op (MODIFIED FOR BIGQUERY - PRODUCTION)


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open(REQUIREMENTS_PATH)
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def evaluate_model_op(
    test_data: Input[dsl.types.BQTable],  # <-- CHANGED
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

    FEATURE_COLUMNS = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]

    logging.basicConfig(level=logging.INFO)
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    table_uri = test_data.metadata.get("destinationTable", test_data.uri)
    logging.info("[PROD] Reading test data from BigQuery: %s", table_uri)
    query = f"SELECT * FROM `{table_uri}`"
    test_df = bq_client.query(query).to_dataframe()
    logging.info("[PROD] Loaded %d test rows from BigQuery", len(test_df))
    model_artifact = joblib.load(model.path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    predictions = model_artifact.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("min_accuracy_threshold", min_accuracy)

    logging.info(
        "[PROD] Accuracy = %.4f",
        accuracy
    )
    return accuracy

# Component: model_approved_op


@component(
    base_image=BASE_IMAGE
)
def model_approved_op(model_accuracy: float, model: Input[Model]):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "[PROD] ✅ Model approved with accuracy: %.4f",
        model_accuracy
    )
    logging.info(
        "[PROD] Ready for registration from: %s",
        model.uri
    )

# Component: register_model_op


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
            "[PROD] Registering new version under parent model: %s",
            parent_model
        )

    model = aiplatform.Model.upload(**upload_args)
    logging.info(
        "[PROD] Model registered: %s",
        model.resource_name
    )

# Component: model_rejected_op


@component(
    base_image=BASE_IMAGE
)
def model_rejected_op(model_accuracy: float, min_accuracy: float):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.error(
        "[PROD] ❌ Model rejected. Accuracy %.4f < %.2f",
        model_accuracy,
        min_accuracy
    )
    raise ValueError(
        "Model accuracy does not meet minimum production threshold."
    )

# Pipeline: prod_diabetes_pipeline (MODIFIED FOR BIGQUERY)


@dsl.pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def prod_diabetes_pipeline(
    project_id: str,
    region: str,
    model_display_name: str,
    bq_dataset: str,
    bq_view: str,
    reg_rate: float = 0.05,
    min_accuracy: float = 0.75,
    parent_model: str = ""
):
    # Query training data from BigQuery view (80% split)
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

    # Query test data from BigQuery view (20% split)
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
