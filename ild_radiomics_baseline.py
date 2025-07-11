import time
import os
import glob
import pandas as pd
from radiomics import featureextractor
from joblib import Parallel, delayed
# from tqdm import tqdm

# # ----------------------------
# Start timing the script
start_time = time.time()
# # ---------------------------

# Paths
ct_dir = ""
mask_dir = ""
output_dir = ""
param_file = ""

# Define where to write the combined CSV
os.makedirs(output_dir, exist_ok=True)
out_csv = os.path.join(output_dir, "all_cases_radiomics.csv")

# Setup
extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
print("Enabled features:", extractor.enabledFeatures)
cases = [os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(mask_dir, "*.nii.gz"))]

def process_case(case):
    img = os.path.join(ct_dir,   f"{case}_0000.nii.gz")
    msk = os.path.join(mask_dir, f"{case}.nii.gz")
    if not os.path.exists(img):
        print(f" CT missing for {case}, skipping")
        return None
    feats = extractor.execute(img, msk)
    feats["p"] = case
    return feats

# Run 8 jobs in parallel (use -1 to use all cores)
results = Parallel(n_jobs=8, backend="loky")(
    delayed(process_case)(case)
    for case in cases
)

# filter out any Nones (skipped cases) and build DataFrame
rows = [r for r in results if r is not None]
df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)

# # ----------------------------
# End timing the script
end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60
print(f"\nTotal execution time: {elapsed_minutes:.2f} minutes")

