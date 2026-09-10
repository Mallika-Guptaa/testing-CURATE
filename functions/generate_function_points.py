def generate_function_points(folder: str, output1: str) -> None:
    import numpy as np
    import pandas as pd
    import tempfile
    import os

    faasr_log("Generating 40 points for y = 20x + 33")

    x = np.arange(1, 41, dtype=float)
    y = 20 * x + 33

    df = pd.DataFrame({"x": x, "y": y})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
        df.to_csv(tmp_path, index=False)

    try:
        faasr_put_file(local_file=tmp_path, remote_folder=folder, remote_file=output1)
        faasr_log(f"Saved {len(df)} points to {output1}")
    finally:
        os.unlink(tmp_path)
