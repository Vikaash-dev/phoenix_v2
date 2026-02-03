# Deploying Phoenix Mamba V2 to Kaggle

The most robust way to deploy your custom library to a Kaggle Kernel is by packaging it as a standard Python wheel (`.whl`) and uploading it as a Kaggle Dataset.

## Prerequisites
- Kaggle API installed and configured (`~/.kaggle/kaggle.json`)
- `build` package installed (`pip install build`)

## Step 1: Build the Package
We have already configured `pyproject.toml`. Run the build command to generate the wheel:

```bash
cd 1_projects/phoenix_mamba_v2
python -m build
# The .whl file will be in dist/
```

## Step 2: Create/Update Kaggle Dataset
We have prepared a `kaggle_dist` directory for this purpose.

1. Copy the wheel to the distribution folder:
   ```bash
   cp dist/*.whl kaggle_dist/
   ```

2. Initialize metadata (if not already done):
   ```bash
   kaggle datasets init -p kaggle_dist
   # Edit kaggle_dist/dataset-metadata.json with your username and dataset slug
   ```
   *Note: A template `dataset-metadata.json` has been created for you in `kaggle_dist/`.*

3. Create the dataset (first time):
   ```bash
   kaggle datasets create -p kaggle_dist
   ```

4. Or update the dataset (subsequent versions):
   ```bash
   kaggle datasets version -p kaggle_dist -m "Updated package version"
   ```

## Step 3: Use in Kaggle Kernel

### Option A: Manual Setup (GUI)
1. Open your Kaggle Notebook.
2. Click **"Add Data"** in the right sidebar.
3. Search for "Phoenix Mamba V2 Library" (or "Your Datasets").
4. Add the dataset.

### Option B: API / Metadata
If creating a kernel via API, add the dataset slug to your `kernel-metadata.json`:

```json
{
  "id": "username/my-brain-tumor-detection-kernel",
  "title": "My Kernel",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "username/phoenix-mamba-v2-lib"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

## Step 4: Install in Notebook
In the first cell of your notebook, install the package from the attached dataset:

```python
# Install the custom library from the dataset
# Note: The path depends on the dataset name. It is usually under /kaggle/input/
!pip install /kaggle/input/phoenix-mamba-v2-lib/phoenix_mamba_v2-2.0.0-py3-none-any.whl --no-deps

# Verify installation
import phoenix_mamba_v2
print("Phoenix Mamba V2 version:", phoenix_mamba_v2.__version__)
```

*Note: We use `--no-deps` if we want to rely on Kaggle's pre-installed environment for things like TensorFlow/Numpy. Remove it if you want pip to try and resolve dependencies (requires Internet).*
