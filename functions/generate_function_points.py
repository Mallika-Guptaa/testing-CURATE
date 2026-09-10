def generate_function_points(folder: str, input1: str, output1: str) -> None:
    """Generate 20 data points for y = 2x + 3 and save as CSV."""
    import pandas as pd
    import json
    import os

    faasr_log("Starting generate_function_points")

    config_local = "workflow_config_local.json"
    faasr_get_file(local_file=config_local, remote_folder=folder, remote_file=input1)

    with open(config_local, "r") as f:
        config = json.load(f)

    faasr_log(f"Loaded config: {config}")

    x_values = list(range(1, 21))
    y_values = [2 * x + 3 for x in x_values]

    df = pd.DataFrame({"x": x_values, "y": y_values})

    local_output = "function_points_local.csv"
    df.to_csv(local_output, index=False)

    faasr_log(f"Generated {len(df)} data points for y = 2x + 3")

    faasr_put_file(local_file=local_output, remote_folder=folder, remote_file=output1)

    if os.path.exists(config_local):
        os.remove(config_local)
    if os.path.exists(local_output):
        os.remove(local_output)

    faasr_log("generate_function_points complete")
