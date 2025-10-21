"""
Loads pre-built Kubeflow Pipelines (KFP) components from Google Cloud.

This module centralizes the loading of all pre-built, externally managed
components used in the MLOps pipelines. This approach follows the DRY
(Don't Repeat Yourself) principle and makes it easy to manage and update
component versions in one location.

All comments and documentation lines are kept <= 100 characters for .flake8.
"""

from kfp import components

# Load the pre-built BigQuery query component from its official URL.
# This component executes a BigQuery query job and returns the results
# as a BQTable artifact.
bigquery_query_job_op = components.load_component_from_url(
    'https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/'
    'bigquery-query-job/sha256:'
    'd1cae80bc0de4e5b95b994739c8d0d7d42ce5a4cb17d3c9512eaed14540f6343'
)