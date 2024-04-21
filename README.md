


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
  
-----------main text about the result report----------
# Optimizing PyTorch DataLoader Performance: The Impact of Worker Count
PyTorch's DataLoader is a powerful tool for efficiently loading and preprocessing data for training deep learning models. However, its performance can be influenced by various factors, including the number of worker processes used during data loading. In this blog post, we'll explore the impact of the worker count on DataLoader performance and provide insights into optimizing its usage.

## Introduction
Data loading is often a bottleneck in the training pipeline of deep learning models. PyTorch's DataLoader addresses this challenge by allowing for parallel data loading using multiple worker processes. By default, PyTorch uses a single worker process, but users can specify a higher number to leverage parallelism and speed up data loading.

## Experimental Setup
To investigate the influence of the worker count on DataLoader performance, we conducted a series of experiments using a custom DataLoader implemented in PyTorch. Our DataLoader is designed for loading medical imaging data for a segmentation task.

We varied two parameters in our experiments:

- Number of Workers: We tested DataLoader performance with 1, 2, 4, and 8 worker processes.
- Batch Size: We evaluated different batch sizes ranging from 1 to 8.

## Results
### Impact on data loading speed
We observed that increasing the number of worker processes generally led to faster data loading and therefore may lead to faster training. However, the magnitude of improvement varied depending on factors such as batch size and hardware configuration.

For smaller batch sizes (e.g., 1 or 2), doubling the number of workers resulted in significant speedups, especially on systems with multiple CPU cores or GPUs. However, as the batch size increased, the marginal improvement from adding more workers diminished.
![image](https://github.com/prerakmody/torch-parallel-dataloading/assets/34941987/268e5968-fe22-4c2c-8446-87f5e04cd738)


![image](https://github.com/prerakmody/torch-parallel-dataloading/assets/34941987/1f67a323-217e-44c5-970f-eaaa61f14b46)

### Resource Utilization
We also monitored resource utilization during data loading with varying worker counts. With a higher number of workers, we observed increased CPU and memory usage, which is expected due to the parallelism introduced by additional processes. Users should consider their hardware constraints and resource availability when choosing the optimal worker count.


## Best Practices and Recommendations
Based on our experiments, we provide the following best practices and recommendations for optimizing DataLoader performance:

- Start with a Baseline: Begin with a conservative number of workers and gradually increase it while monitoring performance metrics.
- Consider Batch Size: Adjust the worker count based on the batch size and hardware capabilities. Higher batch sizes may require fewer workers for optimal performance.
- Monitor Resource Usage: Keep track of CPU, memory, and GPU utilization to ensure efficient resource allocation.
- Test Across Hardware Configurations: Validate DataLoader performance on different hardware configurations to ensure scalability and stability.
## Conclusion
In conclusion, the number of worker processes in PyTorch's DataLoader plays a crucial role in determining data loading and training speeds. By carefully tuning this parameter and considering factors such as batch size and hardware configuration, users can optimize DataLoader performance for their specific use cases. Experimentation and monitoring are key to identifying the optimal worker count and maximizing training efficiency.

Stay tuned for more insights and tips on deep learning optimization!

