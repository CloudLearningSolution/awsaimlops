"""
Vertex AI KFP Pipeline for Development
- Preprocesses data from GCS
- Trains a model in development mode
- Evaluates the model
- Conditionally registers the model in Vertex AI Model Registry if accuracy is sufficient
"""

from kfp import dsl
from kfp.dsl import (
    component,
    pipeline,
    Input,
    Output,
    Dataset,
    Model,
    Metrics
)

PIPELINE_NAME = "mlops-diabetes-dev-pipeline"
PIPELINE_DESCRIPTION = (
    "Development pipeline for diabetes prediction model on Vertex AI"
)

BASE_IMAGE = "python:3.9"
REQUIREMENTS_PATH = "src/requirements.txt"


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open("src/requirements.txt")
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def preprocess_data_op(
    input_gcs_uri: str,
    output_train_data: Output[Dataset],
    output_test_data: Output[Dataset]
):
    import pandas as pd
    from google.cloud import storage
    from urllib.parse import urlparse
    import logging

    logging.basicConfig(level=logging.INFO)
    parsed = urlparse(input_gcs_uri)
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    local_file = "diabetes_raw_dev.csv"
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(
        local_file
    )
    df = pd.read_csv(local_file)

    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)

    train_data.to_csv(output_train_data.path, index=False)
    test_data.to_csv(output_test_data.path, index=False)

    logging.info(
        "[DEV] Preprocessed and split data saved to: %s, %s",
        output_train_data.path,
        output_test_data.path,
    )


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open("src/requirements.txt")
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def train_model_op(
    train_data: Input[Dataset],
    output_model: Output[Model],
    reg_rate: float
):
    import pandas as pd
    import joblib
    from sklearn.linear_model import LogisticRegression
    import logging
    import os
    import shutil

    FEATURE_COLUMNS = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]

    logging.basicConfig(level=logging.INFO)
    train_df = pd.read_csv(train_data.path)
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Diabetic"]

    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X, y)

    # Save as model.joblib for Vertex AI compatibility
    model_path = os.path.join(os.path.dirname(output_model.path), "model.joblib")
    joblib.dump(model, model_path)
    # Also copy to output_model.path for downstream components
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
        for pkg in open("src/requirements.txt")
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def evaluate_model_op(
    test_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics],
    min_accuracy: float
) -> float:
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score
    import logging

    FEATURE_COLUMNS = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]

    logging.basicConfig(level=logging.INFO)
    test_df = pd.read_csv(test_data.path)
    model_artifact = joblib.load(model.path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    predictions = model_artifact.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("min_accuracy_threshold", min_accuracy)

    logging.info(
        "[DEV] Accuracy = %.4f",
        accuracy
    )
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
        for pkg in open("src/requirements.txt")
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
    input_raw_data_gcs_uri: str,
    reg_rate: float = 0.05,
    min_accuracy: float = 0.70,
    parent_model: str = ""
):
    preprocess_task = preprocess_data_op(
        input_gcs_uri=input_raw_data_gcs_uri
    ).set_cpu_limit("1").set_memory_limit("3840Mi")

    train_task = train_model_op(
        train_data=preprocess_task.outputs["output_train_data"],
        reg_rate=reg_rate
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(preprocess_task)

    eval_task = evaluate_model_op(
        model=train_task.outputs["output_model"],
        test_data=preprocess_task.outputs["output_test_data"],
        min_accuracy=min_accuracy
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
