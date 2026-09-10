def generate_points(folder: str, input1: str, output1: str) -> None:
    import pandas as pd

    faasr_log("Generating 20 points for y = 2x + 3")

    faasr_get_file(local_file="workflow-config.json", remote_folder=folder, remote_file=input1)

    x_values = list(range(20))
    y_values = [2 * x + 3 for x in x_values]

    df = pd.DataFrame({'x': x_values, 'y': y_values})

    local_file = "function_points.csv"
    df.to_csv(local_file, index=False)

    faasr_put_file(local_file=local_file, remote_folder=folder, remote_file=output1)

    faasr_log(f"Saved {len(df)} points to {output1}")
