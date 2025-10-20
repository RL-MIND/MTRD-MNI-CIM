# Multi-Teacher Robust Distillation for Memristor-based Computing-in-Memory (MTRD-MNI-CIM)

## Project Overview

This project presents a novel **Multi-Teacher Robust Distillation (MTRD)** training method designed to address non-idealities in **Memristor-based Computing-in-Memory (CIM)** architectures. Memristors are inherently vulnerable to manufacturing-induced variations and operational fluctuations, which cause deviations in programmed weights and compromise the computational accuracy of deep neural networks (DNNs).

Our proposed approach offers a **generalized solution** that is independent of network architecture and task type, making it applicable to a wide range of tasks including **image classification**, **target segmentation**, and **image denoising**. The method significantly mitigates accuracy degradation in memristor-based CIM systems, paving the way for more reliable and scalable deployment in real-world applications.


## Key Contributions

- 🎯 **Generalized Solution**: Independent of network architecture and task type
- 🔧 **Seamless Integration**: Can be integrated into models exhibiting various non-ideal behaviors
- 📊 **Significant Performance**: 33.7% higher accuracy than nominal networks on CIFAR-10 classification
- 🏭 **Hardware Validation**: Verified with fabricated one-transistor-one-memristor (1T1R) chip
- 🌐 **Universal Applicability**: Best performance compared to other variation-aware algorithms

## Task Validation
The pretrained checkpoints for the following three validation tasks can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1JyQzh-b9dRNvtj4S4Emmkkenm1TQbmG1?usp=sharing).

### 1. Image Classification
Deep learning-based image classification tasks designed to evaluate the robustness of MTRD method against memristor non-idealities in classification scenarios.

**Details**: [classification/README.md](./classification/README.md)

### 2. Target Segmentation  
Semantic segmentation tasks for precise target region identification, demonstrating the method's effectiveness in dense prediction problems.

**Details**: [segmentation/README.md](./segmentation/README.md)

### 3. Image Denoising
Image denoising tasks that validate the method's capability in low-level vision problems where precision is crucial for quality enhancement.

**Details**: [denoising/README.md](./denoising/README.md)

## Experimental Results & Hardware Validation

We fabricated a **one-transistor-one-memristor (1T1R) chip** to verify the classification and denoising tasks. The experimental results demonstrate the effectiveness of our MTRD method:

- **Classification Performance**: In a statistical distribution with a standard deviation of 0.5 in weight variations, the accuracy rate was **33.7% higher** than that of nominal networks on CIFAR-10 classification
- **Universality**: Compared to other variation-aware algorithms, our method achieved the **best performance and universality**
- **Hardware Verification**: Successfully validated on fabricated memristor-based hardware, proving practical applicability

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.0 (Recommended)
- Other dependencies listed in `requirements.txt`

## Quick Start

1. **Clone Repository**
   ```bash
   git clone https://github.com/RL-VIG/MTRD-MNI-CIM
   cd MTRD-MNI-CIM
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Specific Tasks**
   
   Navigate to the respective task directories and follow their individual README files:
   - Classification: `cd classification/`
   - Segmentation: `cd segmentation/`
   - Denoising: `cd denoising/`


## Citation

If this project helps your research, please consider citing our work:


## Contact

For questions or suggestions, please contact:
- Email: [zhipingwu@smail.nju.edu]

## License

This project is licensed under the [MIT License](LICENSE).
