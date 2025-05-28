# Contributing to MLX Neural Architecture Search

Thank you for your interest in contributing! This project maintains high scientific standards to ensure reliable, reproducible results for the research community.

## 🔬 Scientific Standards

### Core Principles
1. **No fake metrics** - All measurements must be real and hardware-validated
2. **Statistical rigor** - Use proper confidence intervals and sample sizes
3. **Reproducible methodology** - Document everything clearly
4. **Apple Silicon focus** - Optimizations must be MLX/unified memory specific

### What We DON'T Accept
- ❌ Hardcoded "benchmark" results
- ❌ Estimates masquerading as measurements  
- ❌ CUDA-based approaches ported without validation
- ❌ Pseudoscientific performance claims

### What We DO Want
- ✅ Real benchmarks on actual hardware
- ✅ Statistical validation with confidence intervals
- ✅ Mathematical correctness verification
- ✅ Apple Silicon-specific optimizations

## 🛠 Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/mlx-nas
cd mlx-nas
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Verify scientific correctness
python verify_scientific_rigor.py
```

## 📋 Types of Contributions

### 🔧 Core Components
- **Architecture validators** - New architecture families
- **Component benchmarks** - Additional MLX component testing
- **Search algorithms** - Novel NAS approaches
- **Memory optimizations** - Apple Silicon specific improvements

### 🤖 AI Evolution
- **LLM integrations** - Support for other local models
- **Prompt engineering** - Better architecture generation
- **Learning algorithms** - Improved AI feedback loops
- **Validation methods** - Enhanced architecture scoring

### 🌐 User Interface
- **Visualization improvements** - Better charts and diagrams
- **New pages** - Additional analysis views
- **Export formats** - More data export options
- **Accessibility** - Better UI/UX

### 📚 Documentation
- **Tutorials** - Step-by-step guides
- **Examples** - Real-world use cases
- **API documentation** - Code reference
- **Scientific methodology** - Explaining the rigor

## 🧪 Testing Requirements

### Unit Tests
All new code must include tests:
```python
def test_component_validation():
    """Test that component validation works correctly"""
    # Test mathematical correctness
    # Test statistical properties
    # Test edge cases
```

### Integration Tests
- **End-to-end architecture validation**
- **AI evolution system testing**
- **Web UI functionality**
- **Cross-platform compatibility**

### Scientific Validation
- **Benchmark reproducibility** - Same results across runs
- **Statistical properties** - Proper confidence intervals
- **Mathematical correctness** - Component verification
- **Performance consistency** - Stable measurements

## 📝 Pull Request Process

### 1. Pre-submission Checklist
- [ ] All tests pass
- [ ] Scientific rigor verified
- [ ] Documentation updated
- [ ] Performance benchmarks included
- [ ] Code follows style guidelines

### 2. Scientific Review
Every PR gets reviewed for:
- **Mathematical correctness** - Are the algorithms right?
- **Statistical validity** - Are measurements rigorous?
- **Reproducibility** - Can others get the same results?
- **Apple Silicon optimization** - Is it MLX-specific?

### 3. Code Review
Standard software engineering review:
- Code quality and style
- Test coverage
- Documentation completeness
- API design consistency

## 🎯 Priority Areas

We especially welcome contributions in:

### High Priority
- **Additional architectures** - Vision transformers, MoE, etc.
- **Advanced search methods** - Bayesian optimization, etc.
- **Mobile optimization** - iOS/macOS specific features
- **Educational content** - Tutorials and examples

### Medium Priority
- **UI improvements** - Better visualizations
- **Export formats** - More data formats
- **Integration guides** - Core ML, etc.
- **Performance optimizations** - Speed improvements

### Research Opportunities
- **Novel NAS algorithms** - New search strategies
- **Architecture analysis** - Understanding what works
- **Hardware co-design** - Architecture/chip optimization
- **Efficiency metrics** - Better performance measures

## 🤝 Community Guidelines

### Communication
- **GitHub Issues** - Bug reports and feature requests
- **Discussions** - Questions and general discussion
- **Scientific rigor** - Challenge results respectfully
- **Constructive feedback** - Help improve the science

### Code of Conduct
- Respectful communication
- Focus on scientific merit
- Welcome newcomers
- Share knowledge freely

## 📊 Benchmark Standards

When contributing benchmarks:

### Required Elements
```python
@dataclass
class BenchmarkResult:
    operation_name: str
    mean_time_ms: float
    std_time_ms: float
    confidence_interval_95: Tuple[float, float]
    sample_size: int          # Must be ≥30
    memory_used_mb: float
    hardware_specs: HardwareSpecs
    mlx_version: str
```

### Validation Requirements
- **Minimum 30 samples** for statistical validity
- **Confidence intervals** calculated properly
- **Hardware specifications** documented
- **MLX version** recorded
- **Reproducible setup** described

## 🔬 Research Contributions

For research-oriented contributions:

### Paper Implementations
- **Cite original paper** clearly
- **Validate on MLX** - don't just port
- **Compare fairly** - use same test conditions
- **Document differences** - MLX vs original implementation

### Novel Algorithms
- **Mathematical foundation** - prove correctness
- **Experimental validation** - show it works
- **Comparison baselines** - against existing methods
- **Ablation studies** - understand components

## 🚀 Getting Started

1. **Pick an issue** from our GitHub issues
2. **Comment** that you're working on it
3. **Fork and branch** from main
4. **Implement** with tests and documentation
5. **Verify** scientific rigor
6. **Submit PR** with detailed description

## 💡 Ideas for New Contributors

### Easy First Issues
- Documentation improvements
- Test coverage expansion
- UI polish and bug fixes
- Example notebooks

### Intermediate Projects
- New component implementations
- Additional visualization features
- Performance optimizations
- Export format support

### Advanced Research
- Novel NAS algorithms
- Architecture analysis tools
- Hardware co-design research
- Efficiency metric development

---

**Remember**: This isn't just code - it's scientific infrastructure that researchers and engineers depend on. Every contribution should uphold the highest standards of scientific rigor.

Thank you for helping build the future of Apple Silicon AI! 🚀 