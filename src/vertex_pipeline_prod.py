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

PIPELINE_NAME = "mlops-diabetes-prod-pipeline"
PIPELINE_DESCRIPTION = (
    "Production pipeline for diabetes prediction model on Vertex AI with BigQuery"
)


@pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
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
    # Construct a SQL query to select the training data split (80% of data).
    train_query = f"""
    SELECT *
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(entity_id AS STRING))), 10) < 8
    """
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
    bq_test_task = bigquery_query_job_op(
        project=project_id,
        location=region,
        query=test_query
    )

    train_task = train_model_op(
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region,
        env_prefix="[PROD]",
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(bq_train_task)

    eval_task = evaluate_model_op(
        model=train_task.outputs["output_model"],
        test_data=bq_test_task.outputs["destination_table"],
        min_accuracy=min_accuracy,
        project_id=project_id,
        bq_location=region,
        env_prefix="[PROD]",
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    eval_task.after(train_task)

    with dsl.If(
        eval_task.outputs["Output"] >= min_accuracy,
        name="pass-accuracy-threshold"
    ):
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model=train_task.outputs["output_model"],
            env_prefix="[PROD]",
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        approved_task.after(eval_task)

        register_task = register_model_op(
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            model_artifact=train_task.outputs["output_model"],
            parent_model=parent_model,
            env_prefix="[PROD]",
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        register_task.after(approved_task)

    with dsl.If(
        eval_task.outputs["Output"] < min_accuracy,
        name="fail-accuracy-threshold"
    ):
        rejected_task = model_rejected_op(
            model_accuracy=eval_task.outputs["Output"],
            min_accuracy=min_accuracy,
            env_prefix="[PROD]",
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        rejected_task.after(eval_task)
