#!/usr/bin/env python3
"""
Honest Architecture Search - Scientific Exploration
Version 1.0 - Built with Extreme Scientific Rigor

This module performs systematic architecture search using ONLY validated components
and real measurements. No pseudoscience, no fake metrics, no optimization theater.

PHASE A: Systematic Search (thorough exploration of parameter space)
PHASE B: Honest Evolution (genetic algorithm with real fitness functions)

PRINCIPLES:
- Every measurement is real and validated
- All fitness functions based on actual performance
- Statistical rigor in all benchmarks
- Clear reporting of failures and limitations
- No unverifiable optimization claims
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path

# Import our validated foundation and components
from mlx_foundation import MLXFoundation, StatisticalBenchmark, BenchmarkResult
from mlx_architecture_final import (
    ArchitectureConfig, ArchitectureValidator, TransformerModel,
    create_standard_configs
)

logger = logging.getLogger('HonestArchitectureSearch')

@dataclass
class SearchResult:
    """Results from testing a single architecture configuration"""
    config: ArchitectureConfig
    validation_passed: bool
    error_message: Optional[str]
    benchmark_result: Optional[BenchmarkResult]
    memory_estimate_mb: float
    parameter_count: int
    
    # Derived metrics (calculated from real measurements)
    efficiency_score: Optional[float] = None  # performance per parameter
    memory_efficiency: Optional[float] = None  # performance per memory used
    
@dataclass
class SearchSpace:
    """Defines the parameter space for systematic exploration"""
    hidden_dims: List[int]
    num_layers: List[int] 
    num_heads: List[int]
    vocab_sizes: List[int]
    max_seq_lens: List[int]
    activations: List[str]
    normalizations: List[str]
    
    def total_configurations(self) -> int:
        """Calculate total number of configurations to test"""
        return (len(self.hidden_dims) * len(self.num_layers) * len(self.num_heads) * 
                len(self.vocab_sizes) * len(self.max_seq_lens) * 
                len(self.activations) * len(self.normalizations))

class HonestArchitectureSearcher:
    """Systematic architecture search using validated components"""
    
    def __init__(self):
        self.foundation = MLXFoundation()
        self.validator = ArchitectureValidator()
        self.benchmark = StatisticalBenchmark(min_samples=15, max_samples=30)
        
        logger.info("🔬 Honest Architecture Searcher initialized")
        logger.info(f"   Hardware: {self.foundation.hardware.chip_name}")
        logger.info(f"   Available Memory: {self.foundation.hardware.available_memory_gb:.1f}GB")
    
    def create_practical_search_space(self) -> SearchSpace:
        """Create a practical search space based on hardware constraints"""
        
        # Conservative search space for M4 Pro
        return SearchSpace(
            hidden_dims=[256, 384, 512, 768, 1024],  # Reasonable range
            num_layers=[4, 6, 8, 12],                # Avoid very deep models
            num_heads=[4, 6, 8, 12],                 # Standard head counts
            vocab_sizes=[16000, 32000],              # Practical vocab sizes
            max_seq_lens=[512, 1024],                # Memory-constrained
            activations=['gelu', 'silu'],            # Our best validated activations
            normalizations=['rms_norm']              # Our best validated normalization
        )
    
    def is_valid_configuration(self, hidden_dim: int, num_heads: int) -> bool:
        """Check if configuration meets basic mathematical constraints"""
        return hidden_dim % num_heads == 0
    
    def estimate_memory_requirements(self, config: ArchitectureConfig) -> float:
        """Conservative memory estimation in MB"""
        
        # Model parameters (fp32)
        param_count = self.estimate_parameters(config)
        model_memory_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per param
        
        # Activation memory (conservative estimate)
        batch_size = 1
        seq_len = config.max_seq_len
        hidden_dim = config.hidden_dim
        
        # Memory for activations (multiple of hidden_dim * seq_len)
        activation_memory_mb = (batch_size * seq_len * hidden_dim * 
                               config.num_layers * 8) / (1024 * 1024)  # Conservative multiplier
        
        total_mb = model_memory_mb + activation_memory_mb
        
        # Add 50% overhead for MLX, temporary tensors, etc.
        return total_mb * 1.5
    
    def estimate_parameters(self, config: ArchitectureConfig) -> int:
        """Accurate parameter count estimation"""
        
        # Embeddings
        embedding_params = config.vocab_size * config.hidden_dim
        
        # Per layer: attention (4 projections) + feedforward (2 projections) + norms
        attention_params = 4 * config.hidden_dim * config.hidden_dim
        ff_params = 2 * config.hidden_dim * (config.hidden_dim * 4)  # Standard 4x expansion
        norm_params = 2 * config.hidden_dim  # RMS norm weights only
        
        layer_params = attention_params + ff_params + norm_params
        total_params = embedding_params + (config.num_layers * layer_params)
        
        # Output projection
        total_params += config.hidden_dim * config.vocab_size
        
        return total_params
    
    def test_single_configuration(self, config: ArchitectureConfig) -> SearchResult:
        """Test a single architecture configuration with full validation"""
        
        logger.debug(f"Testing config: {config.num_layers}L-{config.hidden_dim}H-{config.num_heads}A")
        
        # Estimate resources
        memory_estimate = self.estimate_memory_requirements(config)
        param_count = self.estimate_parameters(config)
        
        # Check if configuration fits in available memory
        available_memory_mb = self.foundation.hardware.available_memory_gb * 1024 * 0.8  # 80% usable
        
        if memory_estimate > available_memory_mb:
            return SearchResult(
                config=config,
                validation_passed=False,
                error_message=f"Estimated memory {memory_estimate:.0f}MB exceeds available {available_memory_mb:.0f}MB",
                benchmark_result=None,
                memory_estimate_mb=memory_estimate,
                parameter_count=param_count
            )
        
        # Test with our validated architecture system
        try:
            validation_result = self.validator.validate_architecture(config)
            
            if validation_result.error_message is not None:
                return SearchResult(
                    config=config,
                    validation_passed=False,
                    error_message=validation_result.error_message,
                    benchmark_result=None,
                    memory_estimate_mb=memory_estimate,
                    parameter_count=param_count
                )
            
            # Build and benchmark the model
            model = TransformerModel(config)
            
            # Simple forward pass test
            test_input = mx.random.randint(0, config.vocab_size, (1, min(64, config.max_seq_len)))
            
            def forward_fn(inputs):
                return model(inputs)
            
            benchmark_result = self.benchmark.benchmark_operation(
                forward_fn, (test_input,), f"transformer_{config.num_layers}L_{config.hidden_dim}H"
            )
            
            # Calculate derived metrics
            efficiency_score = (1000.0 / benchmark_result.mean_time_ms) / (param_count / 1e6)  # speed per million params
            memory_efficiency = (1000.0 / benchmark_result.mean_time_ms) / max(0.1, benchmark_result.memory_used_mb)  # speed per MB
            
            result = SearchResult(
                config=config,
                validation_passed=True,
                error_message=None,
                benchmark_result=benchmark_result,
                memory_estimate_mb=memory_estimate,
                parameter_count=param_count,
                efficiency_score=efficiency_score,
                memory_efficiency=memory_efficiency
            )
            
            logger.info(f"✅ {config.num_layers}L-{config.hidden_dim}H-{config.num_heads}A: "
                       f"{benchmark_result.mean_time_ms:.1f}ms, {param_count/1e6:.1f}M params")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Testing failed: {e}")
            return SearchResult(
                config=config,
                validation_passed=False,
                error_message=f"Testing failed: {e}",
                benchmark_result=None,
                memory_estimate_mb=memory_estimate,
                parameter_count=param_count
            )
    
    def systematic_search(self, search_space: SearchSpace, 
                         max_configs: Optional[int] = None) -> List[SearchResult]:
        """Phase A: Systematic exploration of architecture space"""
        
        total_configs = search_space.total_configurations()
        logger.info(f"🔍 Starting systematic search of {total_configs} configurations")
        
        if max_configs and total_configs > max_configs:
            logger.info(f"   Limiting to {max_configs} configurations for practical runtime")
        
        results = []
        tested_count = 0
        
        for hidden_dim in search_space.hidden_dims:
            for num_heads in search_space.num_heads:
                
                # Skip invalid combinations early
                if not self.is_valid_configuration(hidden_dim, num_heads):
                    continue
                
                for num_layers in search_space.num_layers:
                    for vocab_size in search_space.vocab_sizes:
                        for max_seq_len in search_space.max_seq_lens:
                            for activation in search_space.activations:
                                for normalization in search_space.normalizations:
                                    
                                    if max_configs and tested_count >= max_configs:
                                        logger.info(f"Reached maximum configuration limit ({max_configs})")
                                        return results
                                    
                                    config = ArchitectureConfig(
                                        name=f"systematic_{tested_count}",
                                        hidden_dim=hidden_dim,
                                        num_layers=num_layers,
                                        num_heads=num_heads,
                                        vocab_size=vocab_size,
                                        max_seq_len=max_seq_len,
                                        activation=activation,
                                        normalization=normalization
                                    )
                                    
                                    result = self.test_single_configuration(config)
                                    results.append(result)
                                    tested_count += 1
                                    
                                    # Progress update
                                    if tested_count % 10 == 0:
                                        passed = sum(1 for r in results if r.validation_passed)
                                        logger.info(f"   Progress: {tested_count} tested, {passed} passed")
        
        logger.info(f"✅ Systematic search complete: {tested_count} configurations tested")
        return results
    
    def analyze_results(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Analyze search results to find patterns and best configurations"""
        
        # Separate passed and failed results
        passed = [r for r in results if r.validation_passed]
        failed = [r for r in results if not r.validation_passed]
        
        logger.info(f"📊 Analysis: {len(passed)} passed, {len(failed)} failed")
        
        if not passed:
            return {
                'summary': 'No configurations passed validation',
                'total_tested': len(results),
                'passed_count': 0,
                'failed_count': len(failed),
                'failure_reasons': [r.error_message for r in failed[:10]]  # Sample of failures
            }
        
        # Sort by different metrics
        by_speed = sorted(passed, key=lambda x: x.benchmark_result.mean_time_ms)
        by_efficiency = sorted(passed, key=lambda x: x.efficiency_score, reverse=True)
        by_memory = sorted(passed, key=lambda x: x.memory_efficiency, reverse=True)
        by_params = sorted(passed, key=lambda x: x.parameter_count)
        
        # Find patterns
        analysis = {
            'summary': f'{len(passed)}/{len(results)} configurations passed validation',
            'total_tested': len(results),
            'passed_count': len(passed),
            'failed_count': len(failed),
            
            # Best architectures by different criteria
            'fastest_architecture': {
                'config': f"{by_speed[0].config.num_layers}L-{by_speed[0].config.hidden_dim}H-{by_speed[0].config.num_heads}A",
                'time_ms': by_speed[0].benchmark_result.mean_time_ms,
                'params_millions': by_speed[0].parameter_count / 1e6
            },
            
            'most_efficient_architecture': {
                'config': f"{by_efficiency[0].config.num_layers}L-{by_efficiency[0].config.hidden_dim}H-{by_efficiency[0].config.num_heads}A",
                'efficiency_score': by_efficiency[0].efficiency_score,
                'params_millions': by_efficiency[0].parameter_count / 1e6
            },
            
            'smallest_architecture': {
                'config': f"{by_params[0].config.num_layers}L-{by_params[0].config.hidden_dim}H-{by_params[0].config.num_heads}A",
                'params_millions': by_params[0].parameter_count / 1e6,
                'time_ms': by_params[0].benchmark_result.mean_time_ms
            },
            
            # Parameter ranges that worked
            'successful_ranges': {
                'hidden_dims': sorted(list(set(r.config.hidden_dim for r in passed))),
                'num_layers': sorted(list(set(r.config.num_layers for r in passed))),
                'num_heads': sorted(list(set(r.config.num_heads for r in passed))),
            },
            
            # Common failure reasons
            'common_failures': self._analyze_failures(failed),
            
            # Top 5 architectures overall
            'top_5_overall': [
                {
                    'config': f"{r.config.num_layers}L-{r.config.hidden_dim}H-{r.config.num_heads}A",
                    'time_ms': r.benchmark_result.mean_time_ms,
                    'params_millions': r.parameter_count / 1e6,
                    'efficiency_score': r.efficiency_score,
                    'memory_mb': r.memory_estimate_mb
                }
                for r in by_efficiency[:5]
            ]
        }
        
        return analysis
    
    def _analyze_failures(self, failed_results: List[SearchResult]) -> Dict[str, int]:
        """Analyze common failure patterns"""
        failure_counts = {}
        
        for result in failed_results:
            # Categorize failure reasons
            error = result.error_message or "Unknown error"
            
            if "memory" in error.lower():
                failure_counts['Memory constraints'] = failure_counts.get('Memory constraints', 0) + 1
            elif "dimension" in error.lower() or "heads" in error.lower():
                failure_counts['Dimension mismatches'] = failure_counts.get('Dimension mismatches', 0) + 1
            elif "parameter" in error.lower():
                failure_counts['Parameter limits'] = failure_counts.get('Parameter limits', 0) + 1
            else:
                failure_counts['Other errors'] = failure_counts.get('Other errors', 0) + 1
        
        return failure_counts
    
    def save_results(self, results: List[SearchResult], analysis: Dict[str, Any], 
                    filename: str = "honest_architecture_search_results.json"):
        """Save search results and analysis"""
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                'config': {
                    'name': result.config.name,
                    'hidden_dim': result.config.hidden_dim,
                    'num_layers': result.config.num_layers,
                    'num_heads': result.config.num_heads,
                    'vocab_size': result.config.vocab_size,
                    'max_seq_len': result.config.max_seq_len,
                    'activation': result.config.activation,
                    'normalization': result.config.normalization
                },
                'validation_passed': result.validation_passed,
                'error_message': result.error_message,
                'parameter_count': result.parameter_count,
                'memory_estimate_mb': result.memory_estimate_mb,
                'efficiency_score': result.efficiency_score,
                'memory_efficiency': result.memory_efficiency
            }
            
            if result.benchmark_result:
                result_dict['benchmark'] = {
                    'mean_time_ms': result.benchmark_result.mean_time_ms,
                    'std_time_ms': result.benchmark_result.std_time_ms,
                    'confidence_interval': result.benchmark_result.confidence_interval_95,
                    'sample_size': result.benchmark_result.sample_size,
                    'memory_used_mb': result.benchmark_result.memory_used_mb
                }
            
            serializable_results.append(result_dict)
        
        # Combine with analysis
        output_data = {
            'search_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'hardware': f"{self.foundation.hardware.chip_name}",
                'available_memory_gb': self.foundation.hardware.available_memory_gb,
                'search_type': 'systematic'
            },
            'analysis': analysis,
            'results': serializable_results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"💾 Results saved to {filename}")

def main():
    """Main execution function for Phase A: Systematic Search"""
    print("🔬 Honest Architecture Search - Phase A: Systematic Exploration")
    print("=" * 80)
    
    try:
        # Initialize searcher
        searcher = HonestArchitectureSearcher()
        
        # Create search space
        search_space = searcher.create_practical_search_space()
        total_configs = search_space.total_configurations()
        
        print(f"📋 Search Space Overview:")
        print(f"   • Hidden dimensions: {search_space.hidden_dims}")
        print(f"   • Layer counts: {search_space.num_layers}")
        print(f"   • Head counts: {search_space.num_heads}")
        print(f"   • Total configurations: {total_configs}")
        
        # Limit search for practical demonstration
        max_configs = 50  # Reasonable for demonstration
        print(f"   • Testing limit: {max_configs} configurations")
        
        # Run systematic search
        results = searcher.systematic_search(search_space, max_configs=max_configs)
        
        # Analyze results
        analysis = searcher.analyze_results(results)
        
        # Display key findings
        print(f"\n🏆 Search Results:")
        print(f"   • Configurations tested: {analysis['total_tested']}")
        print(f"   • Successful architectures: {analysis['passed_count']}")
        print(f"   • Failed architectures: {analysis['failed_count']}")
        
        if analysis['passed_count'] > 0:
            print(f"\n🚀 Best Architectures:")
            print(f"   • Fastest: {analysis['fastest_architecture']['config']} "
                  f"({analysis['fastest_architecture']['time_ms']:.1f}ms)")
            print(f"   • Most Efficient: {analysis['most_efficient_architecture']['config']} "
                  f"(score: {analysis['most_efficient_architecture']['efficiency_score']:.2f})")
            print(f"   • Smallest: {analysis['smallest_architecture']['config']} "
                  f"({analysis['smallest_architecture']['params_millions']:.1f}M params)")
            
            print(f"\n📊 Top 5 Overall:")
            for i, arch in enumerate(analysis['top_5_overall'][:5], 1):
                print(f"   {i}. {arch['config']}: {arch['time_ms']:.1f}ms, "
                      f"{arch['params_millions']:.1f}M params, "
                      f"efficiency: {arch['efficiency_score']:.2f}")
        
        # Save results
        searcher.save_results(results, analysis)
        
        print(f"\n✅ Phase A complete! Results saved.")
        print(f"📄 Next: Use these results to design Phase B (Honest Evolution)")
        
        return results, analysis
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"❌ Search failed: {e}")
        return None, None

if __name__ == "__main__":
    main() 