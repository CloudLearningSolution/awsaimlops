---
challenge:
    module: 'Deploy a model with GitHub Actions'
    challenge: '6: Deploy and test the model'
---

<style>
.button  {
  border: none;
  color: white;
  padding: 12px 28px;
  background-color: #008CBA;
  float: right;
}
</style>

# Challenge 6: Deploy and test the model

<button class="button" onclick="window.location.href='https://cloud.google.com/vertex-ai/docs/general/deployment';">Google Cloud Vertex AI Documentation</button>

## Challenge scenario

To get value from a model, you'll want to deploy it. You can deploy a model to a managed online or batch endpoint using Google Cloud Vertex AI.

## Prerequisites

If you haven't, complete the [previous challenge](05-environments.md) before you continue.

## Objectives

By completing this challenge, you'll learn how to:

- Register the model with GitHub Actions using Vertex AI Model Registry.
- Deploy the model to a Vertex AI online endpoint with GitHub Actions.
- Test the deployed model.

> **Important!**
> Each challenge is designed to allow you to explore how to implement DevOps principles when working with machine learning models. Some instructions may be intentionally vague, inviting you to think about your own preferred approach. If for example, the instructions ask you to create a Google Cloud project, it's up to you to explore and decide how you want to create it. To make it the best learning experience for you, it's up to you to make it as simple or as challenging as you want.

## Challenge Duration

- **Estimated Time**: 45 minutes

## Instructions

When a model is trained and logged, you can easily register and deploy the model with Vertex AI. After training the model, you want to deploy the model to a real-time endpoint so that it can be consumed by a web app.

### Task 1: Register the model in Vertex AI Model Registry
- Upload your trained model artifacts to Google Cloud Storage (if not already there).
- Register the model in Vertex AI Model Registry from your training job output.
- Use the Vertex AI Python SDK or gcloud CLI to register the model with appropriate metadata and labels.

<details>
<summary>Hint</summary>
<br/>
If your model was trained using the Vertex AI SDK with autologging enabled, the model artifacts are automatically saved to Cloud Storage. You can register the model using the `aiplatform.Model.upload()` method or the `gcloud ai models upload` command.
</details>

### Task 2: Create a GitHub Actions workflow for model deployment
- Create a GitHub Actions workflow that deploys the latest version of the registered model.
- The workflow should create a Vertex AI endpoint and deploy your model to the endpoint using the gcloud CLI.
- Include proper authentication using a service account with the necessary Vertex AI permissions.

<details>
<summary>Hint</summary>
<br/>
Use the `gcloud ai endpoints create` command to create an endpoint, then `gcloud ai endpoints deploy-model` to deploy your model. For MLflow models or models with custom prediction containers, Vertex AI provides prebuilt serving containers that handle inference automatically.
</details>

### Task 3: Test the deployed model
- Test whether the deployed model returns predictions as expected.
- Verify the endpoint is accessible and responding correctly to prediction requests.

<details>
<summary>Hint</summary>
<br/>
You can test the endpoint in the Google Cloud Console under Vertex AI > Endpoints, using the gcloud CLI with `gcloud ai endpoints predict`, or by calling the endpoint REST API directly from applications like Postman or curl.
</details>

Here's some sample data to test your diabetes prediction endpoint with:
```json
{
  "instances": [
    {
      "Pregnancies": 6,
      "PlasmaGlucose": 148,
      "DiastolicBloodPressure": 72,
      "TricepsThickness": 35,
      "SerumInsulin": 0,
      "BMI": 33.6,
      "DiabetesPedigree": 0.627,
      "Age": 50
    }
  ]
}
```

### Sample GitHub Actions Workflow

```yaml
name: Deploy Model to Vertex AI

on:
  workflow_dispatch:
  push:
    branches: [ main ]
    paths: [ 'models/**' ]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1
  MODEL_NAME: diabetes-classifier
  ENDPOINT_NAME: diabetes-prediction-endpoint

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v2
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v2
      with:
        project_id: ${{ env.PROJECT_ID }}

    - name: Create Vertex AI Endpoint
      run: |
        gcloud ai endpoints create \
          --region=${{ env.REGION }} \
          --display-name=${{ env.ENDPOINT_NAME }} \
          --project=${{ env.PROJECT_ID }}

    - name: Get Latest Model Version
      id: get-model
      run: |
        MODEL_ID=$(gcloud ai models list \
          --region=${{ env.REGION }} \
          --filter="displayName=${{ env.MODEL_NAME }}" \
          --format="value(name)" \
          --limit=1)
        echo "model_id=$MODEL_ID" >> $GITHUB_OUTPUT

    - name: Get Endpoint ID
      id: get-endpoint
      run: |
        ENDPOINT_ID=$(gcloud ai endpoints list \
          --region=${{ env.REGION }} \
          --filter="displayName=${{ env.ENDPOINT_NAME }}" \
          --format="value(name)" \
          --limit=1)
        echo "endpoint_id=$ENDPOINT_ID" >> $GITHUB_OUTPUT

    - name: Deploy Model to Endpoint
      run: |
        gcloud ai endpoints deploy-model ${{ steps.get-endpoint.outputs.endpoint_id }} \
          --region=${{ env.REGION }} \
          --model=${{ steps.get-model.outputs.model_id }} \
          --display-name=diabetes-classifier-deployment \
          --min-replica-count=1 \
          --max-replica-count=3 \
          --traffic-split=0=100

    - name: Test Endpoint
      run: |
        echo '{"instances":[{"Pregnancies":6,"PlasmaGlucose":148,"DiastolicBloodPressure":72,"TricepsThickness":35,"SerumInsulin":0,"BMI":33.6,"DiabetesPedigree":0.627,"Age":50}]}' > test_data.json

        gcloud ai endpoints predict ${{ steps.get-endpoint.outputs.endpoint_id }} \
          --region=${{ env.REGION }} \
          --json-request=test_data.json
```

## Success criteria

To complete this challenge successfully, you should be able to show:

- A model registered in the Vertex AI Model Registry.
- A successfully completed GitHub Action that deploys the model to a Vertex AI managed online endpoint.
- Evidence that the endpoint responds correctly to prediction requests.

## Useful resources

- [Vertex AI Model Registry documentation](https://cloud.google.com/vertex-ai/docs/model-registry/introduction)
- [Deploy a model to an endpoint using Vertex AI](https://cloud.google.com/vertex-ai/docs/general/deployment)
- [Deploy MLflow models on Vertex AI](https://cloud.google.com/vertex-ai/docs/model-registry/import-model#import_an_mlflow_model)
- [Vertex AI endpoints REST API reference](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints)
- [gcloud CLI documentation for Vertex AI endpoints](https://cloud.google.com/sdk/gcloud/reference/ai/endpoints)
- [gcloud CLI documentation for Vertex AI models](https://cloud.google.com/sdk/gcloud/reference/ai/models)
- [GitHub Actions for Google Cloud](https://github.com/google-github-actions)
- [Vertex AI Python SDK documentation](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk)
- [Google Cloud MLOps best practices](https://cloud.google.com/vertex-ai/docs/start/introduction-mlops)

## Google Cloud Specific Implementation Notes

**Service Account Setup:** Create a service account with the following IAM roles:
- `roles/aiplatform.user` - For Vertex AI operations
- `roles/storage.objectViewer` - For accessing model artifacts in Cloud Storage
- `roles/artifactregistry.reader` - If using custom containers

**Required APIs:** Ensure these APIs are enabled in your Google Cloud project:
- Vertex AI API (`aiplatform.googleapis.com`)
- Cloud Storage API (`storage.googleapis.com`)
- Cloud Build API (`cloudbuild.googleapis.com`) - If building custom containers

**Regions:** Choose an appropriate region for your Vertex AI resources. Popular choices include:
- `us-central1` (Iowa)
- `us-east1` (South Carolina)
- `europe-west1` (Belgium)
- `asia-southeast1` (Singapore)