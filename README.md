# Video Preprocessing Pipeline

A Python package for efficient preprocessing of large video data (>300 GB), optimized for memory efficiency using dask array as well as time efficiency using JAX and CUDA.

## Features
This package processes high-resolution microscopy videos with:
- Support for ND2/TIF formats and multi-channel 2D/3D images
- Fixed pattern noise computation and subtraction
- Configurable pixel binning for enhanced signal and reduced file size
- Multiple output formats (NRRD, TIF) with maximum intensity projection generation for 3D images


## Installation
### System Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support
- GPU Memory: >= 8GB recommended
- System Memory: >= 32GB recommended
  
### Clone this repository
```bash
# Clone the repository
git clone git@github.com:flavell-lab/private_preprocess_videos.git
cd private_preprocess_videos
```

### Install dependencies using pip
```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# For GPU support
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## User-specified parameters

- `--input_path`: Path to input file (.nd2 or .tif)
- `--output_dir`: Directory for processed outputs
- `--noise_path`: Path to precomputed noise reference file (.tif)
- `--blank_dir`: Directory containing noise reference files to be averaged
- `--chunk_size`: Number of frames to process at once
- `--n_z`: Number of Z-slices per volume
- `--x_range`: X dimension range as "start,end"
- `--y_range`: Y dimension range as "start,end"
- `--z_range`: Z dimension range as "start,end"
- `--channels`: Channels to process as "1,2"
- `--bitdepth`: Bit depth of images
- `--binsize`: Binning factor
- `--save_as`: Output format ('nrrd' or 'tif')
- `--gpu`: GPU device number (optional, only if user wants to dedicate a specific GPU to the current run)

### Processing individual ND2 Files
```bash
python main.py \
    --input_path /path/to/raw/nd2 \
    --output_dir /path/to/processed/output \
    --blank_dir /path/to/blank/frames/directory \
    --chunk_size 64 \
    --n_z 80 \
    --x_range 0,966 \
    --y_range 0,630 \
    --z_range 3,80 \
    --channels 1,2 \
    --bitdepth 12 \
    --binsize 3 \
    --global_t_start 0 \
    --global_t_end 1600 \
    --gpu 2
```

You can also stay with the default parameters for typical 16-minute whole-brain calcium imaging recordings acquired with Flavell Lab's custom hardware and NIS Elements software by omitting all optional arguments, in which case the following operations will be performed:

After 3x3 binning, a uniform background intensity of 800 is assumed for all frames, which will be subtracted from each superpixel. Each frame is then clamped to fit in the range of (0, 4096] as 12-bit and exported as NRRD files. These operations minimize nonlinear transformations in the preprocessing step while preserving variance in pixel values. Processed images are ready to be passed into pretrained neural networks for segmentation and registration.
```bash
python main.py \
    --input_path /store1/shared/panneuralGFP_SWF1212/data_raw/2025-02-06/2025-02-06-01.nd2 \
    --output_dir /store1/shared/panneuralGFP_SWF1212/data_processed/2025-02-06-01_output
```

### Processing multiple ND2 Files at once
Attach a `metadata.txt` in your input directory, and submit multiple nd2 files by running the script `submit_multiple.sh`.
Here is an example of the metadata file, which defines all the immobilized recordings affiliated with each freely moving recording:
```bash
"2025-03-18-21":
{
all_red = "2025-03-18-22"
mNeptune = "2025-03-18-23"
OFP = "2025-03-18-24"
BFP = "2025-03-18-25"
}

"2025-03-18-26":
{
all_red = "2025-03-18-28"
mNeptune = "2025-03-18-30"
OFP = "2025-03-18-31"
BFP = "2025-03-18-32"
}
```

Below is the an example command line to submit all nd2 files acquired on a given day as specified in `metadata.txt` in one go.
Specify which CUDA device you'd like to use with flag `-g`:
```bash
./submit_multiple.sh \
    -i "/store1/shared/panneuralGFP_new/data_raw/2025-03-18" \
    -o "/store1/shared/panneuralGFP_new/data_processed" \
    -g 2
```

This will first print all the commands that call `main.py`, then execute all of them.