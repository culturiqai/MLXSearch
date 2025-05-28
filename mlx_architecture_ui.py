#!/usr/bin/env python3
"""
MLX Architecture UI - Comprehensive Interface
Version 1.0 - Built with Extreme Scientific Rigor

A complete Streamlit interface for the MLX architecture search and AI evolution system.
Integrates all components: foundation analysis, component testing, architecture validation,
systematic search, and AI-driven evolution.

FEATURES:
- Real-time hardware monitoring
- Interactive architecture configuration
- Live progress tracking for searches
- Comprehensive results visualization
- Export/import capabilities
- Scientific reporting
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
import threading
import queue
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import logging
from pathlib import Path

# Import our MLX components
try:
    from mlx_foundation import MLXFoundation, HardwareDetector
    from mlx_components_v2 import ComponentTester
    from mlx_architecture_final import (
        ArchitectureConfig, ArchitectureValidator, create_standard_configs
    )
    from ai_evolution_search import AIEvolutionConfig, AIArchitectureEvolution
    from honest_architecture_search import HonestArchitectureSearcher, SearchSpace
    MLX_AVAILABLE = True
except ImportError as e:
    MLX_AVAILABLE = False
    MLX_ERROR = str(e)

# Configure Streamlit
st.set_page_config(
    page_title="MLX Architecture Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global state management
if 'foundation' not in st.session_state:
    st.session_state.foundation = None
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'current_search' not in st.session_state:
    st.session_state.current_search = None

class UILogger:
    """Custom logger for UI display"""
    
    def __init__(self):
        self.logs = []
        self.max_logs = 100
    
    def log(self, level: str, message: str):
        timestamp = time.strftime('%H:%M:%S')
        self.logs.append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })
        
        # Keep only recent logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
    
    def get_logs(self) -> List[Dict]:
        return self.logs
    
    def clear(self):
        self.logs = []

# Initialize UI logger
if 'ui_logger' not in st.session_state:
    st.session_state.ui_logger = UILogger()

def check_dependencies():
    """Check if all dependencies are available"""
    if not MLX_AVAILABLE:
        st.error(f"❌ MLX components not available: {MLX_ERROR}")
        st.markdown("""
        **Required Dependencies:**
        ```bash
        pip install mlx mlx-lm streamlit plotly
        ```
        """)
        return False
    return True

def initialize_foundation():
    """Initialize MLX foundation with caching"""
    if st.session_state.foundation is None:
        try:
            with st.spinner("Initializing MLX Foundation..."):
                st.session_state.foundation = MLXFoundation()
                st.session_state.ui_logger.log("INFO", "MLX Foundation initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize MLX Foundation: {e}")
            return False
    return True

def sidebar():
    """Create sidebar with navigation and system info"""
    with st.sidebar:
        st.title("🧠 MLX Architecture Lab")
        st.markdown("*Scientific Neural Architecture Search*")
        
        # Navigation
        page = st.selectbox(
            "Navigate",
            [
                "🏠 Dashboard",
                "🔧 Hardware Analysis", 
                "🧪 Component Testing",
                "🏗️ Architecture Builder",
                "🔍 Systematic Search",
                "🤖 AI Evolution",
                "📊 Results Viewer",
                "📝 Logs & Monitoring"
            ]
        )
        
        st.divider()
        
        # System Status
        st.subheader("System Status")
        
        if st.session_state.foundation:
            hardware = st.session_state.foundation.hardware
            
            # Memory status
            memory_used_pct = (1 - hardware.available_memory_gb / hardware.total_memory_gb) * 100
            st.metric(
                "Memory Usage",
                f"{memory_used_pct:.1f}%",
                f"{hardware.available_memory_gb:.1f}GB free"
            )
            
            # CPU info
            st.metric("CPU Cores", hardware.cpu_cores)
            
            # Status indicators
            st.success("✅ MLX Ready")
            
            if 'mlx_lm' in str(st.session_state.foundation):
                st.success("✅ Phi-3 Available")
            else:
                st.warning("⚠️ Phi-3 Not Loaded")
        else:
            st.error("❌ MLX Not Initialized")
        
        st.divider()
        
        # Quick Actions
        st.subheader("Quick Actions")
        
        if st.button("🔄 Refresh System"):
            st.session_state.foundation = None
            st.rerun()
        
        if st.button("🗑️ Clear Logs"):
            st.session_state.ui_logger.clear()
            st.rerun()
        
        if st.button("💾 Export Results"):
            export_results()
    
    return page

def dashboard_page():
    """Main dashboard page"""
    st.title("🏠 MLX Architecture Lab Dashboard")
    
    if not st.session_state.foundation:
        st.warning("Please initialize the system first.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Architectures Tested",
            len(st.session_state.results_history),
            delta=None
        )
    
    with col2:
        successful = sum(1 for r in st.session_state.results_history if r.get('success', False))
        st.metric(
            "Successful Validations",
            successful,
            delta=f"{successful/max(1, len(st.session_state.results_history))*100:.1f}% success rate"
        )
    
    with col3:
        st.metric(
            "Hardware",
            st.session_state.foundation.hardware.chip_name,
            delta=f"{st.session_state.foundation.hardware.total_memory_gb:.0f}GB"
        )
    
    # Recent Activity
    st.subheader("📈 Recent Activity")
    
    if st.session_state.results_history:
        # Create timeline chart
        df = pd.DataFrame(st.session_state.results_history)
        if not df.empty:
            fig = px.timeline(
                df,
                x_start="timestamp",
                x_end="timestamp", 
                y="config_name",
                color="success",
                title="Architecture Testing Timeline"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No architecture tests run yet. Try the Architecture Builder or Systematic Search!")
    
    # Quick Start
    st.subheader("🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏗️ Build Architecture", use_container_width=True):
            st.session_state.page = "🏗️ Architecture Builder"
            st.rerun()
    
    with col2:
        if st.button("🔍 Start Systematic Search", use_container_width=True):
            st.session_state.page = "🔍 Systematic Search"
            st.rerun()
    
    with col3:
        if st.button("🤖 Run AI Evolution", use_container_width=True):
            st.session_state.page = "🤖 AI Evolution"
            st.rerun()

def hardware_analysis_page():
    """Hardware analysis and benchmarking page"""
    st.title("🔧 Hardware Analysis")
    
    if not st.session_state.foundation:
        st.warning("Please initialize the system first.")
        return
    
    hardware = st.session_state.foundation.hardware
    
    # Hardware Overview
    st.subheader("Hardware Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Chip:** {hardware.chip_name}
        **Total Memory:** {hardware.total_memory_gb:.1f} GB
        **Available Memory:** {hardware.available_memory_gb:.1f} GB
        **CPU Cores:** {hardware.cpu_cores}
        """)
    
    with col2:
        # Memory visualization
        memory_data = {
            'Used': hardware.total_memory_gb - hardware.available_memory_gb,
            'Available': hardware.available_memory_gb
        }
        
        fig = px.pie(
            values=list(memory_data.values()),
            names=list(memory_data.keys()),
            title="Memory Usage",
            color_discrete_map={'Used': '#ff7f7f', 'Available': '#7fbf7f'}
        )
        st.plotly_chart(fig)
    
    # Benchmarking
    st.subheader("🏃 Performance Benchmarking")
    
    if st.button("Run Hardware Benchmarks"):
        with st.spinner("Running benchmarks..."):
            try:
                # Run basic operation validation
                operation_results = st.session_state.foundation.validate_basic_operations()
                
                # Display results
                benchmark_data = []
                for name, result in operation_results.items():
                    benchmark_data.append({
                        'Operation': name,
                        'Mean Time (ms)': result.mean_time_ms,
                        'Std Dev (ms)': result.std_time_ms,
                        'Memory (MB)': result.memory_used_mb,
                        'Samples': result.sample_size
                    })
                
                df = pd.DataFrame(benchmark_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualization
                fig = px.bar(
                    df,
                    x='Operation',
                    y='Mean Time (ms)',
                    error_y='Std Dev (ms)',
                    title="MLX Operation Performance"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.ui_logger.log("INFO", "Hardware benchmarks completed")
                
            except Exception as e:
                st.error(f"Benchmarking failed: {e}")

def component_testing_page():
    """Component testing and validation page"""
    st.title("🧪 Component Testing")
    
    if not st.session_state.foundation:
        st.warning("Please initialize the system first.")
        return
    
    st.markdown("Test and validate individual neural network components.")
    
    # Component selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Test Configuration")
        
        hidden_dim = st.selectbox("Hidden Dimension", [128, 256, 384, 512, 768])
        seq_len = st.selectbox("Sequence Length", [64, 128, 256, 512])
        batch_size = st.selectbox("Batch Size", [1, 2, 4])
        
        components_to_test = st.multiselect(
            "Components to Test",
            [
                "Standard Attention",
                "RMS Normalization", 
                "Layer Normalization",
                "ReLU Activation",
                "GELU Activation",
                "SiLU Activation"
            ],
            default=["Standard Attention", "RMS Normalization", "GELU Activation"]
        )
    
    with col2:
        st.subheader("Test Results")
        
        if st.button("🧪 Run Component Tests"):
            if not components_to_test:
                st.warning("Please select at least one component to test.")
                return
            
            with st.spinner("Testing components..."):
                try:
                    tester = ComponentTester()
                    results = {}
                    
                    progress_bar = st.progress(0)
                    
                    for i, component_name in enumerate(components_to_test):
                        st.session_state.ui_logger.log("INFO", f"Testing {component_name}")
                        
                        # Map component names to test functions
                        if component_name == "Standard Attention":
                            from mlx_components_v2 import StandardAttention
                            component = StandardAttention(hidden_dim, hidden_dim // 64)
                            result = tester.test_component(
                                component, 'attention', 'standard_attention',
                                hidden_dim, seq_len, batch_size
                            )
                        elif component_name == "RMS Normalization":
                            from mlx_components_v2 import RMSNorm
                            component = RMSNorm(hidden_dim)
                            result = tester.test_component(
                                component, 'normalization', 'rms_norm',
                                hidden_dim, seq_len, batch_size
                            )
                        # Add other components...
                        
                        results[component_name] = result
                        progress_bar.progress((i + 1) / len(components_to_test))
                    
                    # Display results
                    display_component_results(results)
                    
                except Exception as e:
                    st.error(f"Component testing failed: {e}")

def display_component_results(results: Dict):
    """Display component test results"""
    
    # Results table
    result_data = []
    for name, result in results.items():
        result_data.append({
            'Component': name,
            'Status': '✅ PASS' if result.validation.mathematical_correctness else '❌ FAIL',
            'Performance (ms)': f"{result.performance.mean_time_ms:.2f}" if result.performance else "N/A",
            'Memory (MB)': f"{result.performance.memory_used_mb:.1f}" if result.performance else "N/A",
            'Efficiency': f"{result.memory_efficiency:.2f}" if result.memory_efficiency else "N/A"
        })
    
    df = pd.DataFrame(result_data)
    st.dataframe(df, use_container_width=True)
    
    # Performance visualization
    if any(r.performance for r in results.values()):
        perf_data = {
            name: result.performance.mean_time_ms 
            for name, result in results.items() 
            if result.performance
        }
        
        fig = px.bar(
            x=list(perf_data.keys()),
            y=list(perf_data.values()),
            title="Component Performance Comparison",
            labels={'x': 'Component', 'y': 'Time (ms)'}
        )
        st.plotly_chart(fig, use_container_width=True)

def architecture_builder_page():
    """Interactive architecture builder page"""
    st.title("🏗️ Architecture Builder")
    
    if not st.session_state.foundation:
        st.warning("Please initialize the system first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configure Architecture")
        
        # Architecture parameters
        arch_name = st.text_input("Architecture Name", "custom_architecture")
        hidden_dim = st.selectbox("Hidden Dimension", [128, 256, 384, 512, 768, 1024])
        num_layers = st.selectbox("Number of Layers", [2, 4, 6, 8, 12, 16])
        
        # Ensure num_heads divides hidden_dim
        valid_heads = [h for h in [2, 4, 6, 8, 12, 16] if hidden_dim % h == 0]
        num_heads = st.selectbox("Number of Heads", valid_heads)
        
        vocab_size = st.selectbox("Vocabulary Size", [8000, 16000, 32000, 50000])
        max_seq_len = st.selectbox("Max Sequence Length", [256, 512, 1024, 2048])
        activation = st.selectbox("Activation", ["relu", "gelu", "silu"])
        normalization = st.selectbox("Normalization", ["rms_norm", "layer_norm"])
        
        # Quick presets
        st.subheader("Quick Presets")
        if st.button("📱 Mobile-Optimized"):
            set_preset("mobile")
        if st.button("🖥️ Desktop-Optimized"):
            set_preset("desktop")
        if st.button("🚀 High-Performance"):
            set_preset("performance")
    
    with col2:
        st.subheader("Architecture Preview")
        
        # Create config
        config = ArchitectureConfig(
            name=arch_name,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            activation=activation,
            normalization=normalization
        )
        
        # Display config info
        display_architecture_info(config)
        
        # Test architecture
        if st.button("🧪 Test This Architecture", use_container_width=True):
            test_single_architecture(config)

def display_architecture_info(config: ArchitectureConfig):
    """Display architecture information and estimates"""
    
    # Parameter estimation
    try:
        from honest_architecture_search import HonestArchitectureSearcher
        searcher = HonestArchitectureSearcher()
        param_count = searcher.estimate_parameters(config)
        memory_estimate = searcher.estimate_memory_requirements(config)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Parameters", f"{param_count/1e6:.1f}M")
        
        with col2:
            st.metric("Memory Estimate", f"{memory_estimate:.0f}MB")
        
        with col3:
            # Memory fit status
            available_mb = st.session_state.foundation.hardware.available_memory_gb * 1024 * 0.8
            fits = memory_estimate <= available_mb
            st.metric("Memory Fit", "✅ Yes" if fits else "❌ No")
        
        # Architecture visualization
        st.subheader("Architecture Diagram")
        create_architecture_diagram(config)
        
    except Exception as e:
        st.error(f"Error calculating estimates: {e}")

def create_architecture_diagram(config: ArchitectureConfig):
    """Create a visual diagram of the architecture"""
    
    # Simple block diagram
    blocks = []
    
    # Input embedding
    blocks.append(f"Input Embeddings\n({config.vocab_size:,} × {config.hidden_dim})")
    
    # Transformer blocks
    for i in range(config.num_layers):
        blocks.append(f"Transformer Block {i+1}\n{config.num_heads} heads, {config.activation}")
    
    # Output
    blocks.append(f"Output Projection\n({config.hidden_dim} × {config.vocab_size:,})")
    
    # Create simple flow diagram
    fig = go.Figure()
    
    for i, block in enumerate(blocks):
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[len(blocks) - i],
            mode='markers+text',
            marker=dict(size=100, color='lightblue'),
            text=block,
            textposition='middle center',
            showlegend=False
        ))
    
    fig.update_layout(
        title="Architecture Flow",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def test_single_architecture(config: ArchitectureConfig):
    """Test a single architecture configuration"""
    
    with st.spinner(f"Testing {config.name}..."):
        try:
            validator = ArchitectureValidator()
            result = validator.validate_architecture(config)
            
            # Store result
            result_dict = {
                'timestamp': time.time(),
                'config_name': config.name,
                'config': asdict(config),
                'success': result.error_message is None,
                'error_message': result.error_message,
                'parameter_count': result.parameter_count,
                'memory_mb': result.memory_requirements_mb,
                'forward_pass': result.forward_pass_valid,
                'gradient_flow': result.gradient_flow_valid
            }
            
            st.session_state.results_history.append(result_dict)
            
            # Display results
            if result.error_message is None:
                st.success(f"✅ {config.name} passed all validations!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Parameters", f"{result.parameter_count/1e6:.1f}M")
                with col2:
                    st.metric("Memory", f"{result.memory_requirements_mb:.1f}MB")
                with col3:
                    st.metric("FLOPs/Token", f"{result.theoretical_flops_per_token/1e6:.1f}M")
                
                # Component validation details
                st.subheader("Component Validation Details")
                comp_data = []
                for name, comp_result in result.component_validations.items():
                    comp_data.append({
                        'Component': name,
                        'Mathematical': '✅' if comp_result.validation.mathematical_correctness else '❌',
                        'Numerical': '✅' if comp_result.validation.numerical_stability else '❌',
                        'Performance': f"{comp_result.performance.mean_time_ms:.2f}ms" if comp_result.performance else "N/A"
                    })
                
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
                
            else:
                st.error(f"❌ {config.name} failed validation: {result.error_message}")
            
            st.session_state.ui_logger.log(
                "INFO" if result.error_message is None else "ERROR",
                f"Tested {config.name}: {'SUCCESS' if result.error_message is None else 'FAILED'}"
            )
            
        except Exception as e:
            st.error(f"Testing failed: {e}")
            st.session_state.ui_logger.log("ERROR", f"Testing {config.name} failed: {e}")

def systematic_search_page():
    """Systematic architecture search page"""
    st.title("🔍 Systematic Architecture Search")
    
    if not st.session_state.foundation:
        st.warning("Please initialize the system first.")
        return
    
    st.markdown("Systematically explore the architecture space with scientific rigor.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Search Configuration")
        
        # Search space configuration
        st.write("**Hidden Dimensions**")
        hidden_dims = st.multiselect(
            "Select values",
            [128, 256, 384, 512, 768, 1024],
            default=[256, 512, 768]
        )
        
        st.write("**Number of Layers**")
        num_layers = st.multiselect(
            "Select values",
            [2, 4, 6, 8, 12, 16],
            default=[4, 6, 8]
        )
        
        st.write("**Number of Heads**")
        num_heads = st.multiselect(
            "Select values", 
            [2, 4, 6, 8, 12, 16],
            default=[4, 8, 12]
        )
        
        vocab_sizes = st.multiselect(
            "Vocabulary Sizes",
            [8000, 16000, 32000, 50000],
            default=[32000]
        )
        
        max_seq_lens = st.multiselect(
            "Max Sequence Lengths",
            [256, 512, 1024, 2048],
            default=[1024]
        )
        
        activations = st.multiselect(
            "Activations",
            ["relu", "gelu", "silu"],
            default=["gelu", "silu"]
        )
        
        normalizations = st.multiselect(
            "Normalizations",
            ["rms_norm", "layer_norm"],
            default=["rms_norm"]
        )
        
        # Calculate total configurations
        total_configs = (len(hidden_dims) * len(num_layers) * len(num_heads) * 
                        len(vocab_sizes) * len(max_seq_lens) * 
                        len(activations) * len(normalizations))
        
        st.info(f"Total configurations: {total_configs}")
        
        max_configs = st.number_input(
            "Limit to configurations", 
            min_value=1, 
            max_value=total_configs, 
            value=min(50, total_configs)
        )
    
    with col2:
        st.subheader("Search Progress")
        
        if st.button("🚀 Start Systematic Search", use_container_width=True):
            run_systematic_search(
                hidden_dims, num_layers, num_heads, vocab_sizes, 
                max_seq_lens, activations, normalizations, max_configs
            )

def run_systematic_search(hidden_dims, num_layers, num_heads, vocab_sizes, 
                         max_seq_lens, activations, normalizations, max_configs):
    """Run systematic architecture search"""
    
    try:
        from honest_architecture_search import HonestArchitectureSearcher, SearchSpace
        
        searcher = HonestArchitectureSearcher()
        
        search_space = SearchSpace(
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_sizes=vocab_sizes,
            max_seq_lens=max_seq_lens,
            activations=activations,
            normalizations=normalizations
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        # Run search with progress updates
        with st.spinner("Running systematic search..."):
            results = searcher.systematic_search(search_space, max_configs)
            
            # Analyze results
            analysis = searcher.analyze_results(results)
            
            # Display results
            display_search_results(results, analysis, results_container)
            
            # Save to session state
            search_result = {
                'type': 'systematic',
                'timestamp': time.time(),
                'results': results,
                'analysis': analysis
            }
            st.session_state.results_history.append(search_result)
            
            st.success(f"✅ Systematic search complete! Tested {len(results)} configurations.")
            
    except Exception as e:
        st.error(f"Systematic search failed: {e}")

def ai_evolution_page():
    """AI-driven evolution page"""
    st.title("🤖 AI Evolution Search")
    
    if not st.session_state.foundation:
        st.warning("Please initialize the system first.")
        return
    
    st.markdown("Use local Phi-3 AI to intelligently evolve neural architectures.")
    
    # Check for mlx-lm
    try:
        import mlx_lm
        phi3_available = True
    except ImportError:
        phi3_available = False
        st.error("❌ mlx-lm not installed. Install with: `pip install mlx-lm`")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Evolution Configuration")
        
        max_iterations = st.slider("Max Iterations", 1, 20, 10)
        suggestions_per_iteration = st.slider("Suggestions per Iteration", 1, 10, 4)
        temperature = st.slider("AI Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.slider("Max Tokens", 500, 3000, 1500)
        
        model_path = st.selectbox(
            "Phi-3 Model",
            [
                "mlx-community/Phi-3-mini-4k-instruct-4bit",
                "mlx-community/Phi-3-mini-128k-instruct-4bit"
            ]
        )
        
        # Evolution strategy
        st.subheader("Strategy")
        use_manual_test = st.checkbox("Start with manual test architectures")
        
        if use_manual_test:
            st.info("Will test predefined architectures first to give AI context.")
    
    with col2:
        st.subheader("Evolution Progress")
        
        if st.button("🧠 Start AI Evolution", use_container_width=True):
            run_ai_evolution(
                max_iterations, suggestions_per_iteration, 
                temperature, max_tokens, model_path, use_manual_test
            )

def run_ai_evolution(max_iterations, suggestions_per_iteration, 
                    temperature, max_tokens, model_path, use_manual_test):
    """Run AI-driven evolution"""
    
    try:
        config = AIEvolutionConfig(
            max_iterations=max_iterations,
            suggestions_per_iteration=suggestions_per_iteration,
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        evolution = AIArchitectureEvolution(config)
        
        # Progress tracking
        progress_container = st.container()
        
        with st.spinner("Initializing AI Evolution..."):
            if use_manual_test:
                # Start with manual test
                manual_configs = evolution.create_test_architectures()
                st.info(f"Testing {len(manual_configs)} manual architectures first...")
                
                manual_results = evolution.run_manual_test(manual_configs)
                st.success(f"Manual test complete. Starting AI evolution...")
            
            # Run AI evolution
            results = evolution.run_evolution()
            
            # Display results
            display_ai_evolution_results(results)
            
            # Save to session state
            evolution_result = {
                'type': 'ai_evolution',
                'timestamp': time.time(),
                'results': results
            }
            st.session_state.results_history.append(evolution_result)
            
            st.success("✅ AI Evolution complete!")
            
    except Exception as e:
        st.error(f"AI Evolution failed: {e}")

def display_ai_evolution_results(results: Dict):
    """Display AI evolution results"""
    
    metadata = results['evolution_metadata']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Architectures Tested", metadata['total_architectures_tested'])
    
    with col2:
        st.metric("Successful", metadata['successful_architectures'])
    
    with col3:
        st.metric("Success Rate", f"{metadata['overall_success_rate']:.1%}")
    
    with col4:
        st.metric("Total Time", f"{metadata['total_time_seconds']:.1f}s")
    
    # Best architectures
    if results['best_architectures'] and results['best_architectures']['highest_performance']:
        st.subheader("🏆 Best Architecture Found")
        
        best = results['best_architectures']['highest_performance']
        config_info = best['config']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json(config_info)
        
        with col2:
            st.metric("Performance Score", f"{best['performance_score']:.2f}")
            st.metric("Parameters", f"{best['parameter_count']/1e6:.1f}M")
            st.metric("Memory", f"{best['memory_mb']:.1f}MB")
            st.metric("Efficiency Ratio", f"{best['efficiency_ratio']:.2f}")
    
    # AI performance metrics
    if 'ai_performance_metrics' in results:
        ai_metrics = results['ai_performance_metrics']
        
        st.subheader("🧠 AI Learning Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Generation Time", f"{ai_metrics['average_generation_time_ms']:.1f}ms")
        
        with col2:
            if ai_metrics.get('learning_improvement') is not None:
                improvement = ai_metrics['learning_improvement']
                st.metric("Learning Improvement", f"{improvement:+.1%}")
        
        with col3:
            st.metric(
                "Success Rate Trend",
                f"{ai_metrics['early_success_rate']:.1%} → {ai_metrics['late_success_rate']:.1%}"
            )

def results_viewer_page():
    """Results viewer and comparison page"""
    st.title("📊 Results Viewer")
    
    if not st.session_state.results_history:
        st.info("No results available yet. Run some architecture tests first!")
        return
    
    # Filter and search
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        
        result_types = list(set(r.get('type', 'single') for r in st.session_state.results_history))
        selected_types = st.multiselect("Result Types", result_types, default=result_types)
        
        show_only_successful = st.checkbox("Show only successful")
        
        # Date range
        st.write("**Time Range**")
        hours_back = st.slider("Hours back", 1, 168, 24)
        
    with col2:
        st.subheader("Results")
        
        # Filter results
        filtered_results = filter_results(
            st.session_state.results_history,
            selected_types,
            show_only_successful,
            hours_back
        )
        
        # Display results table
        display_results_table(filtered_results)
        
        # Visualizations
        if filtered_results:
            display_results_visualizations(filtered_results)

def filter_results(results: List[Dict], types: List[str], 
                  only_successful: bool, hours_back: int) -> List[Dict]:
    """Filter results based on criteria"""
    
    current_time = time.time()
    time_threshold = current_time - (hours_back * 3600)
    
    filtered = []
    for result in results:
        # Type filter
        if result.get('type', 'single') not in types:
            continue
        
        # Time filter
        if result.get('timestamp', 0) < time_threshold:
            continue
        
        # Success filter
        if only_successful and not result.get('success', True):
            continue
        
        filtered.append(result)
    
    return filtered

def display_results_table(results: List[Dict]):
    """Display results in a table format"""
    
    if not results:
        st.info("No results match the current filters.")
        return
    
    # Convert to DataFrame
    table_data = []
    for result in results:
        if result.get('type') == 'single':
            table_data.append({
                'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp'])),
                'Name': result['config_name'],
                'Type': result.get('type', 'single'),
                'Status': '✅ Success' if result.get('success') else '❌ Failed',
                'Parameters': f"{result.get('parameter_count', 0)/1e6:.1f}M",
                'Memory': f"{result.get('memory_mb', 0):.1f}MB"
            })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

def display_results_visualizations(results: List[Dict]):
    """Display visualizations of results"""
    
    st.subheader("📈 Visualizations")
    
    # Success rate over time
    success_data = []
    for result in results:
        success_data.append({
            'timestamp': result['timestamp'],
            'success': result.get('success', False)
        })
    
    if success_data:
        df = pd.DataFrame(success_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = px.scatter(
            df,
            x='datetime',
            y='success',
            title="Success Rate Over Time",
            color='success'
        )
        st.plotly_chart(fig, use_container_width=True)

def logs_monitoring_page():
    """Logs and monitoring page"""
    st.title("📝 Logs & Monitoring")
    
    # Real-time logs
    st.subheader("📋 System Logs")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Refresh Logs"):
            st.rerun()
        
        if st.button("🗑️ Clear Logs"):
            st.session_state.ui_logger.clear()
            st.rerun()
        
        # Log level filter
        log_levels = st.multiselect(
            "Log Levels",
            ["INFO", "WARNING", "ERROR"],
            default=["INFO", "WARNING", "ERROR"]
        )
    
    with col1:
        logs = st.session_state.ui_logger.get_logs()
        
        if logs:
            # Filter by level
            filtered_logs = [log for log in logs if log['level'] in log_levels]
            
            # Display logs
            for log in filtered_logs[-50:]:  # Show last 50 logs
                level_emoji = {
                    'INFO': 'ℹ️',
                    'WARNING': '⚠️', 
                    'ERROR': '❌'
                }.get(log['level'], 'ℹ️')
                
                st.text(f"{log['timestamp']} {level_emoji} {log['message']}")
        else:
            st.info("No logs available.")
    
    # System monitoring
    if st.session_state.foundation:
        st.subheader("🖥️ System Monitoring")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Memory usage
            hardware = st.session_state.foundation.hardware
            memory_used_pct = (1 - hardware.available_memory_gb / hardware.total_memory_gb) * 100
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = memory_used_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Memory Usage %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def export_results():
    """Export results to JSON"""
    
    if not st.session_state.results_history:
        st.warning("No results to export.")
        return
    
    # Prepare export data
    export_data = {
        'export_timestamp': time.time(),
        'system_info': {
            'hardware': asdict(st.session_state.foundation.hardware) if st.session_state.foundation else None
        },
        'results': st.session_state.results_history
    }
    
    # Convert to JSON
    json_str = json.dumps(export_data, indent=2, default=str)
    
    # Offer download
    st.download_button(
        label="💾 Download Results JSON",
        data=json_str,
        file_name=f"mlx_architecture_results_{int(time.time())}.json",
        mime="application/json"
    )

def set_preset(preset_name: str):
    """Set architecture preset values"""
    # This would set session state values for the architecture builder
    # Implementation depends on how the form state is managed
    pass

def main():
    """Main UI function"""
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Initialize foundation
    if not initialize_foundation():
        return
    
    # Sidebar navigation
    page = sidebar()
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "🔧 Hardware Analysis":
        hardware_analysis_page()
    elif page == "🧪 Component Testing":
        component_testing_page()
    elif page == "🏗️ Architecture Builder":
        architecture_builder_page()
    elif page == "🔍 Systematic Search":
        systematic_search_page()
    elif page == "🤖 AI Evolution":
        ai_evolution_page()
    elif page == "📊 Results Viewer":
        results_viewer_page()
    elif page == "📝 Logs & Monitoring":
        logs_monitoring_page()

if __name__ == "__main__":
    main() 