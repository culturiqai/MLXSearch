# 🧠 MLX Architecture Lab - Web UI

A comprehensive Streamlit interface for the MLX Neural Architecture Search and AI Evolution system.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements_ui.txt
   ```

2. **Launch the UI**
   ```bash
   python run_ui.py
   ```
   Or directly with Streamlit:
   ```bash
   streamlit run mlx_architecture_ui.py
   ```

3. **Open in Browser**
   - The UI will automatically open at `http://localhost:8501`

## 📋 Features Overview

### 🏠 Dashboard
- **System Overview**: Hardware status, memory usage, test summary
- **Recent Activity**: Timeline of architecture tests and searches
- **Quick Actions**: One-click access to main features

### 🔧 Hardware Analysis
- **Real-time Monitoring**: Memory usage, CPU cores, chip information
- **Performance Benchmarking**: MLX operation speed tests
- **Memory Capacity Analysis**: Maximum model size estimates

### 🧪 Component Testing
- **Individual Component Validation**: Test attention, normalization, activations
- **Mathematical Correctness**: Verify component implementations
- **Performance Benchmarking**: Component-level speed and memory tests
- **Interactive Configuration**: Customize test parameters

### 🏗️ Architecture Builder
- **Interactive Design**: Drag-and-drop style architecture configuration
- **Real-time Estimates**: Parameter count, memory usage, compatibility
- **Quick Presets**: Mobile-optimized, desktop-optimized, high-performance
- **Visual Architecture Diagram**: See your model structure
- **Instant Validation**: Test architectures with one click

### 🔍 Systematic Search
- **Scientific Exploration**: Systematically test architecture combinations
- **Configurable Search Space**: Choose parameters to explore
- **Progress Tracking**: Real-time updates on search progress
- **Results Analysis**: Automatic analysis of successful configurations
- **Failure Pattern Detection**: Understand what doesn't work

### 🤖 AI Evolution
- **Local Phi-3 Integration**: Use local AI for architecture evolution
- **Intelligent Suggestions**: AI learns from previous results
- **Configuration Flexibility**: Adjust AI creativity and iteration count
- **Learning Metrics**: Track AI improvement over iterations
- **Manual Testing Mode**: Start with predefined architectures

### 📊 Results Viewer
- **Comprehensive History**: All test results in one place
- **Advanced Filtering**: Filter by type, success, time range
- **Comparison Tools**: Compare architectures side-by-side
- **Visualizations**: Charts and graphs of performance trends
- **Export Capabilities**: Download results as JSON

### 📝 Logs & Monitoring
- **Real-time Logs**: See what's happening under the hood
- **System Monitoring**: Live memory and CPU usage
- **Error Tracking**: Detailed error messages and debugging info
- **Log Filtering**: Focus on specific types of messages

## 🎯 Usage Workflows

### Workflow 1: Quick Architecture Test
1. Go to **Architecture Builder**
2. Configure your architecture parameters
3. Click **Test This Architecture**
4. View validation results and performance metrics

### Workflow 2: Systematic Exploration
1. Go to **Systematic Search**
2. Select parameter ranges to explore
3. Set maximum configurations to test
4. Click **Start Systematic Search**
5. Review results in **Results Viewer**

### Workflow 3: AI-Driven Evolution
1. Go to **AI Evolution**
2. Configure evolution parameters
3. Optional: Start with manual test architectures
4. Click **Start AI Evolution**
5. Watch AI learn and improve suggestions

### Workflow 4: Component Analysis
1. Go to **Component Testing**
2. Select components to validate
3. Configure test parameters
4. Click **Run Component Tests**
5. Analyze mathematical correctness and performance

## 🔧 Configuration Options

### Architecture Parameters
- **Hidden Dimension**: 128, 256, 384, 512, 768, 1024
- **Number of Layers**: 2, 4, 6, 8, 12, 16
- **Number of Heads**: 2, 4, 6, 8, 12, 16 (must divide hidden_dim)
- **Vocabulary Size**: 8K, 16K, 32K, 50K
- **Max Sequence Length**: 256, 512, 1024, 2048
- **Activation Functions**: ReLU, GELU, SiLU
- **Normalization**: RMS Norm, Layer Norm

### AI Evolution Settings
- **Max Iterations**: 1-20 (number of evolution cycles)
- **Suggestions per Iteration**: 1-10 (architectures per cycle)
- **AI Temperature**: 0.0-1.0 (creativity vs consistency)
- **Max Tokens**: 500-3000 (response length limit)
- **Model**: Phi-3-mini variants

### Search Configuration
- **Search Space**: Select parameter ranges to explore
- **Configuration Limit**: Prevent excessive testing
- **Hardware Constraints**: Automatic memory limit detection

## 📊 Understanding Results

### Architecture Metrics
- **Parameters**: Total trainable parameters (in millions)
- **Memory**: Estimated memory usage (MB)
- **FLOPs/Token**: Theoretical computation per token
- **Performance Score**: Speed/parameter efficiency ratio
- **Efficiency Ratio**: Performance per memory usage

### Validation Status
- **✅ Mathematical Correctness**: Component math is verified
- **✅ Numerical Stability**: Handles extreme inputs safely
- **✅ Forward Pass**: Model can process inputs
- **✅ Gradient Flow**: Model can be trained

### AI Learning Metrics
- **Learning Improvement**: Success rate increase over iterations
- **Generation Time**: How fast AI generates suggestions
- **Success Trend**: Early vs late iteration performance

## 🛠️ Troubleshooting

### Common Issues

**"MLX components not available"**
- Install MLX: `pip install mlx`
- Check Apple Silicon compatibility

**"mlx-lm not installed"**
- Install for AI Evolution: `pip install mlx-lm`
- Required for Phi-3 local AI features

**"Memory constraints" errors**
- Reduce architecture size (hidden_dim, layers)
- Check available memory in Hardware Analysis
- Close other applications

**UI not loading**
- Check port 8501 is available
- Try different port: `streamlit run mlx_architecture_ui.py --server.port 8502`

### Performance Tips

**Speed up searches:**
- Reduce configuration limits
- Use smaller parameter ranges
- Enable hardware monitoring to track usage

**Improve AI evolution:**
- Start with manual test architectures
- Use lower temperature for consistency
- Increase iterations for better learning

## 📁 File Structure

```
MLX-Architecture-Lab/
├── mlx_architecture_ui.py      # Main UI application
├── run_ui.py                   # UI launcher script
├── requirements_ui.txt         # UI dependencies
├── UI_README.md               # This file
├── mlx_foundation.py          # Hardware analysis
├── mlx_components_v2.py       # Component validation
├── mlx_architecture_final.py  # Architecture assembly
├── ai_evolution_search.py     # AI-driven evolution
├── honest_architecture_search.py  # Systematic search
└── demo_ai_evolution.py       # Evolution demo
```

## 🔬 Scientific Rigor

This UI maintains the same scientific standards as the underlying system:

- **No Fake Metrics**: All measurements are real and validated
- **Statistical Rigor**: Proper confidence intervals and sample sizes
- **Mathematical Validation**: Every component is mathematically verified
- **Transparent Reporting**: Clear indication of failures and limitations
- **Reproducible Results**: All parameters and settings are tracked

## 🤝 Contributing

To extend the UI:

1. **Add New Pages**: Create new page functions in `mlx_architecture_ui.py`
2. **Custom Visualizations**: Use Plotly for interactive charts
3. **Additional Metrics**: Extend result tracking in session state
4. **Enhanced Filtering**: Add new filter options in Results Viewer

## 📧 Support

For issues specific to the UI:
- Check the browser console for JavaScript errors
- Verify all dependencies are installed
- Try refreshing the browser
- Check the terminal for Python errors

For MLX or architecture-related issues:
- Review the Logs & Monitoring page
- Check Hardware Analysis for system constraints
- Verify component validation results

---

**🎉 Enjoy exploring neural architectures with scientific rigor!** 