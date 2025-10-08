import sys
import yaml


def check_cpu_limits(pipeline_yaml_path, expected_cpu="1"):
    with open(pipeline_yaml_path, "r") as f:
        pipeline_spec = yaml.safe_load(f)

    problems = []
    templates = pipeline_spec.get("spec", {}).get("templates", [])

    for t in templates:
        name = t.get("name", "<unnamed>")
        container = t.get("container", {})
        resources = container.get("resources", {})
        limits = resources.get("limits", {})
        cpu = limits.get("cpu")

        if cpu is None:
            problems.append(f"❌ Step '{name}' has no CPU limit set.")
        elif str(cpu).strip() != expected_cpu:
            problems.append(f"⚠️ Step '{name}' requests {cpu} CPUs (expected {expected_cpu}).")

    if not problems:
        print("✅ All steps request exactly 1 CPU.")
    else:
        print("\n".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audit_cpu_limits.py path/to/compiled_pipeline.yaml")
        sys.exit(1)

    check_cpu_limits(sys.argv[1])