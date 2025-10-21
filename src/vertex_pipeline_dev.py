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
"""

from kfp import dsl
from kfp.dsl import pipeline

# Import custom components from the centralized custom components module.
from .custom_components import (
    train_model_op,
    evaluate_model_op,
    model_approved_op,
    register_model_op,
    model_rejected_op,
)
# Import pre-built components from the centralized pre-built components module.
from .prebuilt_components import bigquery_query_job_op

PIPELINE_NAME = "mlops-diabetes-dev-pipeline"
PIPELINE_DESCRIPTION = (
    "Development pipeline for diabetes prediction model on Vertex AI with BigQuery"
)

# ------------------------------------------------------------------------------
# Pipeline: dev_diabetes_pipeline
# ------------------------------------------------------------------------------
# Orchestrates all pipeline steps for the development environment.


@pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
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
    # Construct a SQL query to select the training data split (80% of data).
    train_query = f"""
    SELECT *
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(entity_id AS STRING))), 10) < 8
    """
    # Execute the query using the pre-built BigQuery component.
    bq_train_task = bigquery_query_job_op(
        project=project_id,
        location=region,
        query=train_query
    )

    # Construct a SQL query to select the test data split (20% of data).
    test_query = f"""
    SELECT *
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(entity_id AS STRING))), 10) >= 8
    """
    # Execute the query using the pre-built BigQuery component.
    bq_test_task = bigquery_query_job_op(
        project=project_id,
        location=region,
        query=test_query
    )

    # Train a model using the output of the BigQuery training data task.
    train_task = train_model_op(
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region,
        env_prefix="[DEV]",
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(bq_train_task)

    # Evaluate the trained model using the output of the BigQuery test data task.
    eval_task = evaluate_model_op(
        model=train_task.outputs["output_model"],
        test_data=bq_test_task.outputs["destination_table"],
        min_accuracy=min_accuracy,
        project_id=project_id,
        bq_location=region,
        env_prefix="[DEV]",
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    eval_task.after(train_task)

    # Conditionally approve and register the model if it meets the accuracy threshold.
    with dsl.If(
        eval_task.outputs["Output"] >= min_accuracy,
        name="pass-accuracy-threshold"
    ):
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model=train_task.outputs["output_model"],
            env_prefix="[DEV]",
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        approved_task.after(eval_task)

        register_task = register_model_op(
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            model_artifact=train_task.outputs["output_model"],
            parent_model=parent_model,
            env_prefix="[DEV]",
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        register_task.after(approved_task)

    # Conditionally reject the model if it fails to meet the accuracy threshold.
    with dsl.If(
        eval_task.outputs["Output"] < min_accuracy,
        name="fail-accuracy-threshold"
    ):
        rejected_task = model_rejected_op(
            model_accuracy=eval_task.outputs["Output"],
            min_accuracy=min_accuracy,
            env_prefix="[DEV]",
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        rejected_task.after(eval_task)
