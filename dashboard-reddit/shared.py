from pathlib import Path

import pandas as pd

app_dir = Path(__file__).parent
# hate = pd.read_parquet(app_dir / ".." / "data" / "sampled_hate_cleaned_w_topics.parquet")
# post = pd.read_parquet(app_dir / ".." / "data" / "post.parquet")
df = pd.read_parquet(app_dir / ".." / "data" / "glenn_and_sy.parquet")


# hate = pd.read_parquet(app_dir / "sampled_hate_cleaned_w_topics.parquet") #for showcase purposes