


# Installation
1. Make conda environment
    - ```
      conda create -n torch-parallel-dataloading python=3.9
      conda activate torch-parallel-dataloading
      git clone git@github.com:prerakmody/torch-parallel-dataloading.git
      cd torch-parallel-dataloading
      git checkout main
      conda develop . # adds the current directory to the conda environment path
      ```
2. Install torch v1.12.1
    - `conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge`
3. Install other dependencies
    - `pip install pynrrd scipy scikit-image pandas matplotlib tqdm psutil`

# To test
1. Run the following command
    - `python src/dataloaderCompare.py`