"""
Deploy a Registered Model from Vertex AI Model Registry to an Endpoint

This script provides a complete workflow for deploying a trained and registered
diabetes prediction model to a Vertex AI endpoint for online inference serving.

Compatible with:
- google-cloud-aiplatform==1.56.0
- pandas==1.5.3
- numpy==1.23.5

Main Objectives:
1. Retrieve a registered model from Vertex AI Model Registry
2. Create or use an existing Vertex AI endpoint
3. Deploy the model to the endpoint with appropriate resource allocation
4. Test the deployed endpoint with sample diabetes prediction data
5. Provide monitoring and management capabilities

The script handles both new deployments and updates to existing endpoints,
with proper error handling and logging throughout the process.
"""

import argparse
import json
import logging
import time
from typing import Dict, List, Optional, Tuple

from google.cloud import aiplatform
import pandas as pd


class ModelDeploymentManager:
    """
    Manages the deployment of registered models to Vertex AI endpoints.

    This class encapsulates all functionality needed to:
    - Find and validate registered models
    - Create or reuse endpoints
    - Deploy models with proper configuration
    - Test deployments with sample data
    """

    def __init__(self, project_id: str, region: str):
        """
        Initialize the deployment manager with GCP project details.

        Args:
            project_id: Google Cloud project ID
            region: GCP region for Vertex AI resources (e.g., 'us-central1')
        """
        self.project_id = project_id
        self.region = region

        # Initialize Vertex AI SDK with project and region
        aiplatform.init(project=project_id, location=region)

        # Configure logging for detailed operation tracking
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def find_latest_model(self, model_display_name: str) -> aiplatform.Model:
        """
        Locate the most recently registered model by display name.

        This method searches the Vertex AI Model Registry for models matching
        the specified display name and returns the latest version based on
        creation timestamp.

        Args:
            model_display_name: Display name of the registered model to find

        Returns:
            aiplatform.Model: The latest version of the specified model

        Raises:
            ValueError: If no models found with the specified name
        """
        self.logger.info(f"Searching for model: {model_display_name}")

        # Query all models and filter by display name
        models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')

        if not models:
            raise ValueError(
                f"No models found with display name: {model_display_name}. "
                f"Ensure the model is registered in the Model Registry."
            )

        # Sort by creation time to get the latest version
        latest_model = max(models, key=lambda m: m.create_time)

        self.logger.info(
            f"Found latest model: {latest_model.display_name} "
            f"(Resource: {latest_model.resource_name}, "
            f"Created: {latest_model.create_time})"
        )

        return latest_model

    def create_or_get_endpoint(self, endpoint_display_name: str) -> aiplatform.Endpoint:
        """
        Create a new endpoint or retrieve an existing one by name.

        This method implements an idempotent endpoint creation strategy:
        - If an endpoint with the specified name exists, it returns that endpoint
        - If no endpoint exists, it creates a new one

        Args:
            endpoint_display_name: Display name for the endpoint

        Returns:
            aiplatform.Endpoint: Either the existing or newly created endpoint
        """
        self.logger.info(f"Looking for existing endpoint: {endpoint_display_name}")

        # Search for existing endpoints with the specified name
        existing_endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"'
        )

        if existing_endpoints:
            endpoint = existing_endpoints[0]
            self.logger.info(f"Using existing endpoint: {endpoint.resource_name}")
            return endpoint

        # Create new endpoint if none exists
        self.logger.info(f"Creating new endpoint: {endpoint_display_name}")

        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            description=f"Endpoint for {endpoint_display_name} diabetes prediction model",
            labels={
                "model_type": "diabetes_classifier",
                "environment": "production",
                "managed_by": "vertex_ai_pipeline"
            }
        )

        self.logger.info(f"Created endpoint: {endpoint.resource_name}")
        return endpoint

    def deploy_model_to_endpoint(
        self,
        model: aiplatform.Model,
        endpoint: aiplatform.Endpoint,
        deployment_name: str,
        machine_type: str = "n1-standard-2",
        min_replica_count: int = 1,
        max_replica_count: int = 3,
        traffic_percentage: int = 100
    ) -> Dict:
        """
        Deploy a registered model to a Vertex AI endpoint with specified configuration.

        This method handles the complete deployment process including:
        - Resource allocation and scaling configuration
        - Traffic splitting for gradual rollouts
        - Health checks and deployment validation

        Args:
            model: The registered model to deploy
            endpoint: Target endpoint for deployment
            deployment_name: Name for this specific deployment
            machine_type: Compute instance type (default: n1-standard-2)
            min_replica_count: Minimum number of replicas (default: 1)
            max_replica_count: Maximum number of replicas for auto-scaling (default: 3)
            traffic_percentage: Percentage of traffic to route to this deployment (default: 100)

        Returns:
            Dict: Deployment details including resource allocation and status
        """
        self.logger.info(f"Deploying model {model.display_name} to endpoint {endpoint.display_name}")

        # Configure deployment parameters for production readiness
        self.logger.info("Starting model deployment...")
        deployed_model = endpoint.deploy(
            model=model,
            deployed_model_display_name=deployment_name,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=traffic_percentage,
            sync=True,  # Wait for deployment to complete
        )

        # Validate deployment success
        if deployed_model:
            self.logger.info("Model deployment completed successfully")

            deployment_details = {
                "endpoint_id": endpoint.name,
                "deployed_model_id": deployed_model.id,
                "machine_type": machine_type,
                "replica_count": f"{min_replica_count}-{max_replica_count}",
                "traffic_percentage": traffic_percentage,
                "status": "deployed"
            }

            return deployment_details
        else:
            raise RuntimeError("Model deployment failed - no deployed model returned")

    def test_endpoint_prediction(
        self,
        endpoint: aiplatform.Endpoint,
        test_instances: List[Dict]
    ) -> Tuple[List, bool]:
        """
        Test the deployed endpoint with sample diabetes prediction data.

        This method validates that the endpoint is responding correctly by:
        - Sending sample prediction requests
        - Validating response format and content
        - Checking prediction confidence scores

        Args:
            endpoint: The endpoint to test
            test_instances: List of test data instances for prediction

        Returns:
            Tuple[List, bool]: (prediction_results, test_passed)
        """
        self.logger.info("Testing endpoint with sample diabetes data...")

        try:
            # Send prediction request to the endpoint
            predictions = endpoint.predict(instances=test_instances)

            # Validate prediction results
            if predictions and predictions.predictions:
                self.logger.info("Endpoint responding correctly")

                # Log sample predictions for verification
                for i, prediction in enumerate(predictions.predictions[:3]):  # Show first 3
                    self.logger.info(f"Sample prediction {i+1}: {prediction}")

                return predictions.predictions, True
            else:
                self.logger.error("Endpoint returned empty predictions")
                return [], False

        except Exception as e:
            self.logger.error(f"Endpoint test failed: {str(e)}")
            return [], False

    def get_endpoint_info(self, endpoint: aiplatform.Endpoint) -> Dict:
        """
        Retrieve comprehensive information about the deployed endpoint.

        This method provides detailed endpoint metadata including:
        - Deployment status and health
        - Resource allocation and scaling configuration
        - Traffic distribution across model versions

        Args:
            endpoint: The endpoint to inspect

        Returns:
            Dict: Comprehensive endpoint information
        """
        self.logger.info(f"Retrieving endpoint information: {endpoint.display_name}")

        endpoint_info = {
            "endpoint_name": endpoint.display_name,
            "endpoint_id": endpoint.name,
            "create_time": str(endpoint.create_time),
            "region": self.region,
            "project_id": self.project_id,
            "deployed_models": []
        }

        # Get information about all deployed models on this endpoint
        # Note: Using endpoint._gca_resource.deployed_models for compatibility with aiplatform==1.56.0
        for deployed_model in endpoint._gca_resource.deployed_models:
            model_info = {
                "model_display_name": deployed_model.display_name,
                "model_id": deployed_model.model,
                "machine_type": deployed_model.dedicated_resources.machine_spec.machine_type,
                "min_replicas": deployed_model.dedicated_resources.min_replica_count,
                "max_replicas": deployed_model.dedicated_resources.max_replica_count,
                "traffic_percentage": deployed_model.traffic_percentage
            }
            endpoint_info["deployed_models"].append(model_info)

        return endpoint_info


def create_sample_test_data() -> List[Dict]:
    """
    Generate sample diabetes prediction test data for endpoint validation.

    This function creates realistic test instances that match the expected
    input format for the diabetes prediction model based on the feature columns
    defined in your pipeline files.

    Feature columns from your pipelines:
    - Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness
    - SerumInsulin, BMI, DiabetesPedigree, Age

    Returns:
        List[Dict]: Sample test instances for model prediction
    """
    sample_data = [
        {
            "Pregnancies": 6,
            "PlasmaGlucose": 148,
            "DiastolicBloodPressure": 72,
            "TricepsThickness": 35,
            "SerumInsulin": 0,
            "BMI": 33.6,
            "DiabetesPedigree": 0.627,
            "Age": 50
        },
        {
            "Pregnancies": 1,
            "PlasmaGlucose": 85,
            "DiastolicBloodPressure": 66,
            "TricepsThickness": 29,
            "SerumInsulin": 0,
            "BMI": 26.6,
            "DiabetesPedigree": 0.351,
            "Age": 31
        },
        {
            "Pregnancies": 8,
            "PlasmaGlucose": 183,
            "DiastolicBloodPressure": 64,
            "TricepsThickness": 0,
            "SerumInsulin": 0,
            "BMI": 23.3,
            "DiabetesPedigree": 0.672,
            "Age": 32
        }
    ]

    return sample_data


def main():
    """
    Main execution function that orchestrates the complete model deployment process.

    This function coordinates all deployment steps:
    1. Parse command-line arguments
    2. Initialize the deployment manager
    3. Find the latest registered model
    4. Create or retrieve the target endpoint
    5. Deploy the model with specified configuration
    6. Test the deployment with sample data
    7. Display deployment summary and next steps
    """
    parser = argparse.ArgumentParser(
        description="Deploy a registered Vertex AI model to an endpoint"
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Google Cloud project ID"
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Google Cloud region (e.g., us-central1)"
    )
    parser.add_argument(
        "--model-display-name",
        required=True,
        help="Display name of the registered model to deploy"
    )
    parser.add_argument(
        "--endpoint-display-name",
        required=True,
        help="Display name for the endpoint (created if doesn't exist)"
    )
    parser.add_argument(
        "--deployment-name",
        default="diabetes-model-deployment",
        help="Name for this model deployment"
    )
    parser.add_argument(
        "--machine-type",
        default="n1-standard-2",
        help="Machine type for deployment"
    )
    parser.add_argument(
        "--min-replicas",
        type=int,
        default=1,
        help="Minimum number of replicas"
    )
    parser.add_argument(
        "--max-replicas",
        type=int,
        default=3,
        help="Maximum number of replicas"
    )
    parser.add_argument(
        "--test-endpoint",
        action="store_true",
        help="Test the endpoint after deployment"
    )

    args = parser.parse_args()

    try:
        # Initialize deployment manager
        deployment_manager = ModelDeploymentManager(
            project_id=args.project_id,
            region=args.region
        )

        # Step 1: Find the latest registered model
        model = deployment_manager.find_latest_model(args.model_display_name)

        # Step 2: Create or get the target endpoint
        endpoint = deployment_manager.create_or_get_endpoint(args.endpoint_display_name)

        # Step 3: Deploy the model to the endpoint
        deployment_details = deployment_manager.deploy_model_to_endpoint(
            model=model,
            endpoint=endpoint,
            deployment_name=args.deployment_name,
            machine_type=args.machine_type,
            min_replica_count=args.min_replicas,
            max_replica_count=args.max_replicas
        )

        print("\n" + "="*60)
        print("🎉 MODEL DEPLOYMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Model: {model.display_name}")
        print(f"Endpoint: {endpoint.display_name}")
        print(f"Machine Type: {deployment_details['machine_type']}")
        print(f"Replica Range: {deployment_details['replica_count']}")
        print(f"Traffic Allocation: {deployment_details['traffic_percentage']}%")

        # Step 4: Test the endpoint if requested
        if args.test_endpoint:
            print("\n" + "-"*40)
            print("TESTING ENDPOINT")
            print("-"*40)

            test_data = create_sample_test_data()
            predictions, test_passed = deployment_manager.test_endpoint_prediction(
                endpoint, test_data
            )

            if test_passed:
                print("✅ Endpoint test passed successfully")
                print("Sample predictions:")
                for i, pred in enumerate(predictions[:2]):
                    print(f"  Test {i+1}: {pred}")
            else:
                print("❌ Endpoint test failed")

        # Step 5: Display endpoint information
        endpoint_info = deployment_manager.get_endpoint_info(endpoint)

        print("\n" + "-"*40)
        print("ENDPOINT INFORMATION")
        print("-"*40)
        print(f"Endpoint URL: {endpoint.resource_name}")
        print(f"Total Deployed Models: {len(endpoint_info['deployed_models'])}")

        print("\n" + "-"*40)
        print("NEXT STEPS")
        print("-"*40)
        print("1. Monitor endpoint performance in the Vertex AI console")
        print("2. Set up alerting for endpoint health and latency")
        print("3. Configure auto-scaling based on traffic patterns")
        print("4. Implement A/B testing for model versions")
        print(f"5. Use this endpoint for predictions at: {endpoint.resource_name}")

    except Exception as e:
        print(f"\n❌ Deployment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()