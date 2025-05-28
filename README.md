# 🧠 MLX Neural Architecture Search

**Scientifically rigorous neural architecture search for Apple Silicon, powered by local AI evolution.**

[![MLX](https://img.shields.io/badge/MLX-Compatible-blue)](https://github.com/ml-explore/mlx)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🎯 Overview

The first comprehensive Neural Architecture Search (NAS) system designed specifically for Apple Silicon hardware. Unlike existing tools that rely on fake metrics or transfer learning from CUDA, this system:

- **🔬 Measures real performance** on your actual hardware
- **🤖 Uses local AI** (Phi-3) to intelligently evolve architectures  
- **📊 Provides statistical rigor** with confidence intervals
- **🌐 Includes complete web UI** for interactive exploration
- **⚡ Optimizes for unified memory** architecture of Apple Silicon

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mlx-nas.git
cd mlx-nas

# Install core dependencies
pip install -r requirements.txt

# For web UI (optional)
pip install -r requirements_ui.txt
```

### Basic Usage

```python
from mlx_foundation import MLXFoundation
from mlx_architecture_final import ArchitectureValidator
from mlx_components_v2 import ComponentTester

# Initialize foundation analysis
foundation = MLXFoundation()
report = foundation.generate_foundation_report()

# Test components
tester = ComponentTester()
results = tester.run_comprehensive_validation()

# Validate architectures
validator = ArchitectureValidator(foundation.hardware)
result = validator.validate_architecture(your_config)
```

### Web Interface

```bash
# Launch interactive web UI
python run_ui.py
```

### Examples

```bash
# Run demonstration examples
python examples/demo_ai_evolution.py
python examples/test_evolution_approaches.py
```

## 🏗️ Architecture

The system consists of several scientific modules:

### Core Components

- **`mlx_foundation.py`** - Hardware detection and benchmarking
- **`mlx_components_v2.py`** - Neural network component validation
- **`mlx_architecture_final.py`** - Complete architecture assembly and testing

### Search Methods

- **`honest_architecture_search.py`** - Systematic parameter space exploration
- **`honest_evolution_search.py`** - Genetic algorithm-based search
- **`ai_evolution_search.py`** - Local AI-driven evolution (Phi-3)

### User Interface

- **`mlx_architecture_ui.py`** - Complete Streamlit web interface
- **`run_ui.py`** - UI launcher with dependency checking

## 🧪 Scientific Methodology

### Validation Approach

1. **Real Hardware Measurements** - No synthetic benchmarks
2. **Statistical Rigor** - Proper confidence intervals and sample sizes
3. **Comprehensive Testing** - Mathematical correctness verification
4. **Memory Tracking** - Actual MLX memory usage monitoring

### Architecture Validation

Each architecture undergoes:
- Parameter counting verification
- Forward pass validation
- Gradient flow testing  
- Memory consumption measurement
- Performance benchmarking

## 📊 Features

### 🔬 **Foundation Analysis**
- Accurate Apple Silicon hardware detection
- Statistical benchmarking with confidence intervals
- Memory capacity estimation
- MLX operation validation

### 🧪 **Component Testing**
- Attention mechanism validation
- Normalization layer testing
- Activation function verification
- Mathematical correctness checks

### 🏗️ **Architecture Search**
- Systematic parameter space exploration
- Genetic algorithm evolution
- AI-driven architecture generation
- Multi-objective optimization

### 🌐 **Web Interface**
- Real-time system monitoring
- Interactive architecture designer
- Progress tracking and visualization
- Results comparison and export

## 🎯 Use Cases

### **Researchers**
- Validate new architecture ideas on Apple Silicon
- Compare performance across different designs
- Export results for publications

### **ML Engineers** 
- Find optimal architectures for deployment constraints
- Balance performance vs. memory usage
- Automate architecture optimization

### **Students & Educators**
- Understand architecture trade-offs through visualization
- Learn neural architecture principles hands-on
- Experiment with different design choices

## 📋 Requirements

### Hardware
- Apple Silicon Mac (M1/M2/M3 series)
- 16GB+ unified memory recommended
- macOS 12.0+

### Software
- Python 3.9+
- MLX framework
- See `requirements.txt` for complete list

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our scientific standards and development process.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/mlx-nas.git
cd mlx-nas
pip install -e .

# Run tests
python -m pytest tests/

# Run examples
python examples/demo_ai_evolution.py
```

## 📚 Documentation

- **[Contributing Guidelines](CONTRIBUTING.md)** - Development and contribution process
- **[UI Guide](UI_README.md)** - Complete web interface documentation
- **Examples** - See `examples/` directory for demonstrations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Apple MLX team for the exceptional framework
- The open source ML community
- All contributors and users of this project

## 📞 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Documentation**: Check our comprehensive documentation

---

**⚠️ Note**: This project provides real measurements, not estimates. All performance claims are verified on actual hardware. 