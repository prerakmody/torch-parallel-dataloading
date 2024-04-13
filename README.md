


# Installation
1. Make conda environment
    - ```
      conda create -n interactive-refinement python=3.9
      conda activate interactive-refinement
      git clone git@git.lumc.nl:image-registration/human-centric-ai/hcai-interactive-models.git
      cd hcai-interactive-models
      git checkout main
      conda develop . # adds the current directory to the conda environment path
      ```
2. Install torch v1.12.1
    - `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`
3. Install other dependencies
    - `pip install pynrrd scipy scikit-image matplotlib tqdm psutil`

# To-Do
1. Ignore small errors in patient samples

# Notes
1. Checkout Nvidia DALI

