"""
Submits a Vertex AI Pipeline Job using the Vertex AI SDK for Python.

This script is the recommended, officially supported method for running
Vertex AI Pipelines from a CI/CD environment. It replaces the need for
the gcloud CLI for pipeline submission.

Usage:
    python src/run_pipeline.py --project-id <PROJECT_ID> --region <REGION>
        --pipeline-spec-uri <PIPELINE_SPEC_PATH> --service-account <SERVICE_ACCOUNT>
        --pipeline-root <PIPELINE_ROOT> --display-name <DISPLAY_NAME>
        --parameter-values-json <PARAMS_JSON> --labels-json <LABELS_JSON>
        [--enable-caching]

Arguments:
    --project-id: Google Cloud project ID.
    --region: Google Cloud region for Vertex AI resources.
    --pipeline-spec-uri: Path or GCS URI to the compiled pipeline spec file.
    --service-account: Service account email for running the pipeline job.
    --pipeline-root: GCS path for pipeline output artifacts.
    --display-name: Display name for the pipeline job.
    --parameter-values-json: JSON string of pipeline parameter values.
    --labels-json: JSON string of labels for the pipeline job.
    --enable-caching: Optional flag to enable pipeline step caching.

This script is intended to be used in CI/CD workflows and supports all
required arguments for robust pipeline job submission. Logging is enabled
for all major steps and errors.
"""

import argparse
import json
import logging
from datetime import datetime
from google.cloud import aiplatform


def main():
    # Set up argument parser for all required pipeline job parameters.
    parser = argparse.ArgumentParser(
        description="Submit a Vertex AI Pipeline Job."
    )
    parser.add_argument(
        "--project-id", type=str, required=True, help="Google Cloud project ID."
    )
    parser.add_argument(
        "--region", type=str, required=True, help="Google Cloud region."
    )
    parser.add_argument(
        "--pipeline-spec-uri", type=str, required=True,
        help="GCS URI or local path of the compiled pipeline spec."
    )
    parser.add_argument(
        "--service-account", type=str, required=True,
        help="Service account for the pipeline run."
    )
    parser.add_argument(
        "--pipeline-root", type=str, required=True,
        help="GCS root path for pipeline outputs."
    )
    parser.add_argument(
        "--display-name", type=str, required=True,
        help="Display name for the pipeline run."
    )
    parser.add_argument(
        "--parameter-values-json", type=str, required=True,
        help="JSON string of pipeline parameter values."
    )
    parser.add_argument(
        "--labels-json", type=str, required=True,
        help="JSON string of labels for the pipeline run."
    )
    parser.add_argument(
        "--enable-caching", action="store_true",
        help="Flag to enable caching for the pipeline run."
    )

    args = parser.parse_args()

    # Configure logging for info and error messages.
    logging.basicConfig(level=logging.INFO)
    logging.info("Initializing Vertex AI Platform...")

    # Initialize Vertex AI SDK with project and region.
    aiplatform.init(project=args.project_id, location=args.region)

    # Parse JSON strings for parameter values and labels.
    try:
        parameter_values = json.loads(args.parameter_values_json)
        labels = json.loads(args.labels_json)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON arguments: {e}")
        raise

    # Generate a unique job ID using display name and timestamp.
    job_id = f"{args.display_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logging.info(f"Submitting pipeline job: {job_id}")

    # Create the PipelineJob object with all required parameters.
    pipeline_job = aiplatform.PipelineJob(
        display_name=job_id,
        template_path=args.pipeline_spec_uri,
        pipeline_root=args.pipeline_root,
        parameter_values=parameter_values,
        enable_caching=args.enable_caching,
        labels=labels
    )

    # Submit the pipeline job using the specified service account.
    pipeline_job.submit(service_account=args.service_account)

    # Log success and provide dashboard URI for monitoring.
    logging.info(f"✅ Successfully submitted pipeline job: {job_id}")
    logging.info(f"View in console: {pipeline_job._dashboard_uri()}")


if __name__ == "__main__":
    # Entry point for script execution.
    main()