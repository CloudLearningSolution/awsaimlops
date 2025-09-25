"""
Compiles Kubeflow Pipelines (KFP v2) into YAML pipeline specs for Vertex AI.
Supports dev and prod variants of the diabetes prediction pipeline with dynamic
selection based on the provided Python file name.

This script is intended for use in CI/CD workflows to automate pipeline
compilation for Vertex AI. It ensures that the correct pipeline function is
selected and compiled to a YAML spec for submission.
"""

import argparse
import sys
from typing import Callable
from kfp import compiler

# Import pipeline functions for dev and prod environments.
from vertex_pipeline_dev import dev_diabetes_pipeline
from vertex_pipeline_prod import prod_diabetes_pipeline


def main() -> None:
    """
    Parses command-line arguments and compiles the selected pipeline into a YAML
    file suitable for Vertex AI Pipelines.

    Arguments:
        --py: Path to the Python file defining the pipeline.
        --output: Path to the output compiled YAML file.

    The script selects the pipeline function based on the filename provided in
    --py. If the filename contains 'vertex_pipeline_dev.py', it uses the dev
    pipeline. If it contains 'vertex_pipeline_prod.py', it uses the prod
    pipeline. If neither, it raises a ValueError.

    Compilation errors are caught and printed to stderr. The script exits with
    status code 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Compile a KFP v2 pipeline for Vertex AI."
    )
    parser.add_argument(
        "--py",
        type=str,
        required=True,
        help="Path to the Python file defining the pipeline."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output compiled YAML file."
    )
    args = parser.parse_args()

    # Select pipeline function based on filename.
    if "vertex_pipeline_dev.py" in args.py:
        pipeline_func = dev_diabetes_pipeline
    elif "vertex_pipeline_prod.py" in args.py:
        pipeline_func = prod_diabetes_pipeline
    else:
        # Raise error if filename does not match expected dev/prod pipeline files.
        raise ValueError(
            f"Unknown pipeline file specified: {args.py}. "
            "Must contain 'vertex_pipeline_dev.py' or "
            "'vertex_pipeline_prod.py'."
        )

    try:
        # Compile the selected pipeline function to the specified YAML output.
        compiler.Compiler().compile(
            pipeline_func=pipeline_func,
            package_path=args.output
        )
        print(f"✅ Successfully compiled '{args.py}' → '{args.output}'")
    except Exception as e:
        # Print compilation errors to stderr and exit with status code 1.
        print(f"❌ Failed to compile pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Entry point for script execution.
    main()
