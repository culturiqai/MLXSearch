#!/usr/bin/env python3
"""
AI Evolution Search - Local LLM-Based Architecture Evolution
Version 1.0 - Built with Extreme Scientific Rigor

This module implements scientifically rigorous architecture evolution using
a local Phi-3 model via MLX. Replaces the flawed genetic algorithm with
intelligent AI-driven exploration.

PRINCIPLES:
- Local AI eliminates API costs and latency
- Every suggestion is validated with real benchmarks
- Statistical rigor in all measurements
- Clear learning from previous results
- No pseudoscience or optimization theater
"""

import mlx.core as mx
import mlx.nn as nn
import json
import re
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# Import our validated foundation and components
from mlx_foundation import MLXFoundation, StatisticalBenchmark
from mlx_architecture_final import (
    ArchitectureConfig, ArchitectureValidator, TransformerModel,
    ArchitectureValidation
)

logger = logging.getLogger('AIEvolution')

@dataclass
class AIEvolutionConfig:
    """Configuration for AI-based evolution"""
    max_iterations: int = 15
    suggestions_per_iteration: int = 5
    max_history_context: int = 20  # Architectures to include in LLM context
    model_path: str = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    temperature: float = 0.3  # Balance creativity vs consistency
    max_tokens: int = 2000

@dataclass
class EvolutionResult:
    """Results from testing a single AI-suggested architecture"""
    config: ArchitectureConfig
    validation_result: ArchitectureValidation
    suggestion_source: str  # Which iteration/prompt generated this
    generation_time_ms: float
    
    # Derived metrics for learning
    success: bool
    performance_score: Optional[float] = None
    efficiency_ratio: Optional[float] = None

class LocalPhi3Interface:
    """Interface to Phi-3 model running locally via MLX"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Phi-3 model with proper error handling"""
        try:
            logger.info(f"Loading Phi-3 model: {self.model_path}")
            
            # Import MLX LM modules
            try:
                from mlx_lm import load, generate
                self.generate_fn = generate
                self.model, self.tokenizer = load(self.model_path)
                logger.info("✅ Phi-3 model loaded successfully")
            except ImportError:
                logger.error("❌ mlx-lm not installed. Install with: pip install mlx-lm")
                raise RuntimeError("mlx-lm required for local LLM support")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Phi-3 model: {e}")
            raise RuntimeError(f"Cannot initialize Phi-3: {e}")
    
    def generate_response(self, prompt: str, temperature: float = 0.3, 
                         max_tokens: int = 1000) -> str:
        """Generate response from Phi-3 with error handling"""
        try:
            start_time = time.perf_counter()
            
            # Generate response (mlx-lm doesn't support temperature directly in generate)
            response = self.generate_fn(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
            
            generation_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"LLM generation took {generation_time:.1f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

class ArchitecturePromptGenerator:
    """Generates scientifically informed prompts for architecture evolution"""
    
    def __init__(self):
        self.base_constraints = {
            'hidden_dims': [128, 256, 384, 512, 768, 1024],
            'num_layers': [2, 4, 6, 8, 12, 16],
            'num_heads': [2, 4, 6, 8, 12, 16],
            'vocab_sizes': [8000, 16000, 32000, 50000],
            'max_seq_lens': [256, 512, 1024, 2048],
            'activations': ['relu', 'gelu', 'silu'],
            'normalizations': ['rms_norm', 'layer_norm']
        }
    
    def generate_evolution_prompt(self, successful_configs: List[EvolutionResult],
                                failed_configs: List[EvolutionResult],
                                iteration: int) -> str:
        """Generate scientifically informed prompt for architecture evolution"""
        
        prompt = f"""TASK: You are a neural network architect. Design exactly 3 new transformer configurations.

CRITICAL INSTRUCTIONS:
1. RESPOND WITH ONLY A JSON ARRAY - NO OTHER TEXT
2. NO EXPLANATIONS, NO COMMENTS, NO MARKDOWN
3. JUST THE RAW JSON ARRAY STARTING WITH [ AND ENDING WITH ]

HARDWARE CONSTRAINTS:
- Apple Silicon M4 Pro with 24GB memory
- hidden_dim must be divisible by num_heads

EMPIRICAL EVIDENCE:
{self._format_evidence_brief(successful_configs, failed_configs)}

VALID PARAMETERS:
- hidden_dim: {self.base_constraints['hidden_dims']}
- num_layers: {self.base_constraints['num_layers']}
- num_heads: {self.base_constraints['num_heads']}
- vocab_size: {self.base_constraints['vocab_sizes']}
- max_seq_len: {self.base_constraints['max_seq_lens']}
- activation: {self.base_constraints['activations']}
- normalization: {self.base_constraints['normalizations']}

REQUIRED OUTPUT FORMAT (NO OTHER TEXT):
[
  {{
    "name": "ai_evolved_iter{iteration}_arch1",
    "hidden_dim": 384,
    "num_layers": 6,
    "num_heads": 6,
    "vocab_size": 32000,
    "max_seq_len": 1024,
    "activation": "gelu",
    "normalization": "rms_norm"
  }},
  {{
    "name": "ai_evolved_iter{iteration}_arch2",
    "hidden_dim": 512,
    "num_layers": 8,
    "num_heads": 8,
    "vocab_size": 32000,
    "max_seq_len": 1024,
    "activation": "silu",
    "normalization": "layer_norm"
  }},
  {{
    "name": "ai_evolved_iter{iteration}_arch3",
    "hidden_dim": 768,
    "num_layers": 12,
    "num_heads": 12,
    "vocab_size": 50000,
    "max_seq_len": 2048,
    "activation": "gelu",
    "normalization": "rms_norm"
  }}
]"""

        return prompt
    
    def _format_evidence_brief(self, successful: List[EvolutionResult], 
                              failed: List[EvolutionResult]) -> str:
        """Format evidence in a brief, concise format"""
        evidence = []
        
        if successful:
            evidence.append("SUCCESSFUL PATTERNS:")
            for result in successful[-5:]:  # Last 5 successful
                config = result.config
                evidence.append(f"  ✅ {config.num_layers}L-{config.hidden_dim}H-{config.num_heads}A")
        
        if failed:
            evidence.append("AVOID THESE FAILURES:")
            for result in failed[-3:]:  # Last 3 failures
                config = result.config
                error = result.validation_result.error_message or "Unknown"
                evidence.append(f"  ❌ {config.num_layers}L-{config.hidden_dim}H-{config.num_heads}A: {error[:50]}")
        
        if not evidence:
            evidence.append("NO PREVIOUS DATA - EXPLORE DIFFERENT CONFIGURATIONS")
        
        return "\n".join(evidence)

class ArchitectureParser:
    """Robust parser for LLM-generated architecture configurations"""
    
    def parse_architectures(self, llm_response: str) -> List[ArchitectureConfig]:
        """Parse LLM response into valid architecture configurations with robust extraction"""
        
        try:
            # Multiple strategies to extract JSON from LLM response
            configs = []
            
            # Strategy 1: Find JSON array with flexible regex
            json_patterns = [
                r'\[\s*\{.*?\}\s*\]',  # Standard array
                r'\[[\s\S]*?\]',       # Any array with any content
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, llm_response, re.DOTALL)
                for match in matches:
                    json_text = match.group(0)
                    
                    # Try to clean and parse this JSON
                    parsed_configs = self._try_parse_json_text(json_text)
                    if parsed_configs:
                        configs.extend(parsed_configs)
                        break
                
                if configs:
                    break
            
            # Strategy 2: Line-by-line extraction if no arrays found
            if not configs:
                configs = self._extract_from_lines(llm_response)
            
            # Strategy 3: Manual pattern matching as last resort
            if not configs:
                configs = self._extract_with_patterns(llm_response)
            
            logger.info(f"Successfully extracted {len(configs)} architecture configurations")
            return configs
            
        except Exception as e:
            logger.error(f"Architecture parsing failed completely: {e}")
            return []
    
    def _try_parse_json_text(self, json_text: str) -> List[ArchitectureConfig]:
        """Try to parse a piece of text as JSON with various cleanup strategies"""
        
        cleanup_strategies = [
            lambda x: x,  # No cleanup
            lambda x: x.strip(),  # Basic strip
            lambda x: re.sub(r'[^\[\]{}",:\w\s-]', '', x),  # Remove invalid chars
            lambda x: re.sub(r'//.*?\n', '\n', x),  # Remove comments
            lambda x: re.sub(r'/\*.*?\*/', '', x, flags=re.DOTALL),  # Remove block comments
        ]
        
        for cleanup in cleanup_strategies:
            try:
                cleaned = cleanup(json_text)
                
                # Handle both array and single object formats
                if cleaned.strip().startswith('['):
                    raw_configs = json.loads(cleaned)
                elif cleaned.strip().startswith('{'):
                    raw_configs = [json.loads(cleaned)]
                else:
                    continue
                
                if not isinstance(raw_configs, list):
                    raw_configs = [raw_configs]
                
                # Validate and convert each configuration
                valid_configs = []
                for i, raw_config in enumerate(raw_configs):
                    config = self._validate_and_create_config(raw_config, i)
                    if config:
                        valid_configs.append(config)
                
                if valid_configs:
                    return valid_configs
                    
            except (json.JSONDecodeError, ValueError):
                continue
        
        return []
    
    def _extract_from_lines(self, text: str) -> List[ArchitectureConfig]:
        """Extract configurations by parsing individual lines for key-value pairs"""
        
        configs = []
        current_config = {}
        
        required_fields = ['hidden_dim', 'num_layers', 'num_heads', 'vocab_size', 
                          'max_seq_len', 'activation', 'normalization']
        
        for line in text.split('\n'):
            line = line.strip()
            
            # Look for key-value patterns
            for field in required_fields:
                patterns = [
                    rf'"{field}":\s*(\d+|"[^"]*")',
                    rf'{field}:\s*(\d+|[^\s,\]}}]+)',
                    rf'{field}[=:]\s*(\d+|[^\s,\]}}]+)'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip('"')
                        try:
                            # Try to convert to int if it looks like a number
                            if value.isdigit():
                                current_config[field] = int(value)
                            else:
                                current_config[field] = value
                        except:
                            pass
            
            # If we have all required fields, create a config
            if len(current_config) >= len(required_fields):
                config = self._validate_and_create_config(current_config, len(configs))
                if config:
                    configs.append(config)
                current_config = {}
        
        return configs
    
    def _extract_with_patterns(self, text: str) -> List[ArchitectureConfig]:
        """Last resort: extract using manual patterns"""
        
        # Look for common architecture descriptions
        arch_patterns = [
            r'(\d+)L[- ](\d+)H[- ](\d+)A',  # "6L-512H-8A" format
            r'layers?\s*[:=]\s*(\d+).*?hidden.*?[:=]\s*(\d+).*?heads?\s*[:=]\s*(\d+)',
        ]
        
        configs = []
        
        for pattern in arch_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for i, match in enumerate(matches):
                try:
                    if len(match.groups()) >= 3:
                        layers, hidden, heads = match.groups()[:3]
                        
                        # Create a basic config with defaults
                        config_dict = {
                            'name': f'pattern_extracted_{i}',
                            'num_layers': int(layers),
                            'hidden_dim': int(hidden),
                            'num_heads': int(heads),
                            'vocab_size': 32000,  # Default
                            'max_seq_len': 1024,  # Default
                            'activation': 'gelu',  # Default
                            'normalization': 'rms_norm'  # Default
                        }
                        
                        config = self._validate_and_create_config(config_dict, i)
                        if config:
                            configs.append(config)
                            
                except (ValueError, IndexError):
                    continue
        
        return configs
    
    def _validate_and_create_config(self, raw_config: Dict, index: int) -> Optional[ArchitectureConfig]:
        """Validate and create a single architecture configuration"""
        
        required_fields = ['hidden_dim', 'num_layers', 'num_heads', 'vocab_size', 
                          'max_seq_len', 'activation', 'normalization']
        
        # Check required fields
        for field in required_fields:
            if field not in raw_config:
                logger.warning(f"Missing required field '{field}' in config {index}")
                return None
        
        # Validate mathematical constraints
        hidden_dim = raw_config['hidden_dim']
        num_heads = raw_config['num_heads']
        
        if hidden_dim % num_heads != 0:
            logger.warning(f"Invalid config {index}: hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
            return None
        
        # Create configuration
        try:
            config = ArchitectureConfig(
                name=raw_config.get('name', f'ai_generated_{index}'),
                hidden_dim=int(hidden_dim),
                num_layers=int(raw_config['num_layers']),
                num_heads=int(num_heads),
                vocab_size=int(raw_config['vocab_size']),
                max_seq_len=int(raw_config['max_seq_len']),
                activation=str(raw_config['activation']),
                normalization=str(raw_config['normalization'])
            )
            
            return config
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to create config {index}: {e}")
            return None

class AIArchitectureEvolution:
    """Main class for AI-driven architecture evolution"""
    
    def __init__(self, config: AIEvolutionConfig = AIEvolutionConfig()):
        self.config = config
        self.validator = ArchitectureValidator()
        self.llm = LocalPhi3Interface(config.model_path)
        self.prompt_generator = ArchitecturePromptGenerator()
        self.parser = ArchitectureParser()
        
        # Evolution state
        self.iteration = 0
        self.successful_results: List[EvolutionResult] = []
        self.failed_results: List[EvolutionResult] = []
        self.all_results: List[EvolutionResult] = []
        
        logger.info("🤖 AI Architecture Evolution initialized")
        logger.info(f"   Model: {config.model_path}")
        logger.info(f"   Max iterations: {config.max_iterations}")
    
    def run_evolution(self) -> Dict[str, Any]:
        """Run complete AI-driven architecture evolution"""
        
        logger.info("🚀 Starting AI Architecture Evolution")
        start_time = time.time()
        
        while self.iteration < self.config.max_iterations:
            logger.info(f"🔄 Iteration {self.iteration + 1}/{self.config.max_iterations}")
            
            # Generate new architectures using AI
            suggested_configs = self._generate_ai_suggestions()
            
            if not suggested_configs:
                logger.warning(f"No valid suggestions in iteration {self.iteration}")
                self.iteration += 1
                continue
            
            # Test each suggested architecture
            iteration_results = []
            for config in suggested_configs:
                result = self._test_architecture(config, f"iter_{self.iteration}")
                iteration_results.append(result)
                self.all_results.append(result)
                
                if result.success:
                    self.successful_results.append(result)
                    logger.info(f"✅ Success: {config.name} - {result.performance_score:.2f}")
                else:
                    self.failed_results.append(result)
                    logger.info(f"❌ Failed: {config.name}")
            
            # Log iteration summary
            iteration_success_rate = sum(1 for r in iteration_results if r.success) / len(iteration_results)
            logger.info(f"📊 Iteration {self.iteration} summary: {iteration_success_rate:.1%} success rate")
            
            self.iteration += 1
        
        # Generate final analysis
        total_time = time.time() - start_time
        analysis = self._analyze_evolution_results(total_time)
        
        # Save results
        self._save_results(analysis)
        
        logger.info("🏁 AI Evolution complete!")
        return analysis
    
    def _generate_ai_suggestions(self) -> List[ArchitectureConfig]:
        """Generate architecture suggestions using Phi-3"""
        
        # Create context-aware prompt
        prompt = self.prompt_generator.generate_evolution_prompt(
            successful_configs=self.successful_results[-self.config.max_history_context:],
            failed_configs=self.failed_results[-self.config.max_history_context:],
            iteration=self.iteration
        )
        
        # Generate response from Phi-3
        logger.debug("🧠 Generating AI suggestions...")
        response = self.llm.generate_response(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        if not response:
            logger.error("Empty response from LLM")
            return []
        
        # Parse architectures from response
        configs = self.parser.parse_architectures(response)
        
        logger.info(f"🎯 Generated {len(configs)} valid architecture suggestions")
        return configs
    
    def _test_architecture(self, config: ArchitectureConfig, source: str) -> EvolutionResult:
        """Test a single architecture with comprehensive validation"""
        
        start_time = time.perf_counter()
        
        try:
            # Validate architecture
            validation_result = self.validator.validate_architecture(config)
            
            generation_time = (time.perf_counter() - start_time) * 1000
            
            # Determine success and calculate performance metrics
            success = validation_result.error_message is None
            performance_score = None
            efficiency_ratio = None
            
            if success:
                # Calculate performance score from validation results
                # Use component validation times as proxy (better than genetic algorithm's approach)
                component_times = []
                for comp_name, comp_result in validation_result.component_validations.items():
                    if comp_result.performance and comp_result.performance.mean_time_ms:
                        component_times.append(comp_result.performance.mean_time_ms)
                
                if component_times:
                    avg_component_time = np.mean(component_times)
                    # Performance score: higher is better (inverse of time, normalized by parameters)
                    performance_score = 1000.0 / (avg_component_time * (validation_result.parameter_count / 1e6))
                    efficiency_ratio = performance_score / validation_result.memory_requirements_mb
            
            result = EvolutionResult(
                config=config,
                validation_result=validation_result,
                suggestion_source=source,
                generation_time_ms=generation_time,
                success=success,
                performance_score=performance_score,
                efficiency_ratio=efficiency_ratio
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Testing failed for {config.name}: {e}")
            
            # Create failed result
            from mlx_architecture_final import ArchitectureValidation
            failed_validation = ArchitectureValidation(
                config=config,
                component_validations={},
                forward_pass_valid=False,
                gradient_flow_valid=False,
                memory_requirements_mb=0.0,
                parameter_count=0,
                theoretical_flops_per_token=0,
                error_message=f"Testing exception: {e}"
            )
            
            return EvolutionResult(
                config=config,
                validation_result=failed_validation,
                suggestion_source=source,
                generation_time_ms=0.0,
                success=False
            )
    
    def _analyze_evolution_results(self, total_time: float) -> Dict[str, Any]:
        """Analyze complete evolution results"""
        
        successful = self.successful_results
        failed = self.failed_results
        
        analysis = {
            'evolution_metadata': {
                'total_time_seconds': total_time,
                'iterations_completed': self.iteration,
                'total_architectures_tested': len(self.all_results),
                'successful_architectures': len(successful),
                'failed_architectures': len(failed),
                'overall_success_rate': len(successful) / len(self.all_results) if self.all_results else 0.0
            },
            
            'best_architectures': self._get_best_architectures(successful),
            'evolution_trends': self._analyze_trends(),
            'failure_analysis': self._analyze_failures(failed),
            'ai_performance_metrics': self._analyze_ai_performance()
        }
        
        return analysis
    
    def _get_best_architectures(self, successful: List[EvolutionResult]) -> List[Dict]:
        """Get top performing architectures"""
        
        if not successful:
            return []
        
        # Sort by performance score
        by_performance = sorted(successful, key=lambda x: x.performance_score or 0, reverse=True)
        by_efficiency = sorted(successful, key=lambda x: x.efficiency_ratio or 0, reverse=True)
        by_parameters = sorted(successful, key=lambda x: x.validation_result.parameter_count)
        
        best = {
            'highest_performance': self._result_to_dict(by_performance[0]) if by_performance else None,
            'most_efficient': self._result_to_dict(by_efficiency[0]) if by_efficiency else None,
            'smallest_model': self._result_to_dict(by_parameters[0]) if by_parameters else None,
            'top_5_overall': [self._result_to_dict(r) for r in by_performance[:5]]
        }
        
        return best
    
    def _result_to_dict(self, result: EvolutionResult) -> Dict:
        """Convert evolution result to dictionary"""
        return {
            'config': asdict(result.config),
            'performance_score': result.performance_score,
            'efficiency_ratio': result.efficiency_ratio,
            'parameter_count': result.validation_result.parameter_count,
            'memory_mb': result.validation_result.memory_requirements_mb,
            'source': result.suggestion_source
        }
    
    def _analyze_trends(self) -> Dict:
        """Analyze trends across iterations"""
        if not self.all_results:
            return {}
        
        # Group by iteration
        by_iteration = {}
        for result in self.all_results:
            iteration = result.suggestion_source
            if iteration not in by_iteration:
                by_iteration[iteration] = []
            by_iteration[iteration].append(result)
        
        # Calculate success rate per iteration
        trends = {}
        for iteration, results in by_iteration.items():
            success_rate = sum(1 for r in results if r.success) / len(results)
            avg_performance = np.mean([r.performance_score for r in results if r.performance_score])
            
            trends[iteration] = {
                'success_rate': success_rate,
                'average_performance': float(avg_performance) if not np.isnan(avg_performance) else None,
                'architectures_tested': len(results)
            }
        
        return trends
    
    def _analyze_failures(self, failed: List[EvolutionResult]) -> Dict:
        """Analyze failure patterns"""
        if not failed:
            return {'total_failures': 0}
        
        failure_reasons = {}
        for result in failed:
            error = result.validation_result.error_message or "Unknown error"
            
            # Categorize failures
            if 'memory' in error.lower():
                category = 'Memory constraints'
            elif 'dimension' in error.lower() or 'heads' in error.lower():
                category = 'Dimension mismatches'
            elif 'parameter' in error.lower():
                category = 'Parameter issues'
            else:
                category = 'Other errors'
            
            failure_reasons[category] = failure_reasons.get(category, 0) + 1
        
        return {
            'total_failures': len(failed),
            'failure_categories': failure_reasons,
            'failure_rate_by_iteration': {}  # Could be expanded
        }
    
    def _analyze_ai_performance(self) -> Dict:
        """Analyze how well the AI performed at suggesting architectures"""
        
        if not self.all_results:
            return {}
        
        # Calculate average generation time
        avg_generation_time = np.mean([r.generation_time_ms for r in self.all_results])
        
        # Learning trend: success rate improvement over time
        early_results = self.all_results[:len(self.all_results)//2] if len(self.all_results) > 6 else []
        late_results = self.all_results[len(self.all_results)//2:] if len(self.all_results) > 6 else []
        
        early_success = sum(1 for r in early_results if r.success) / len(early_results) if early_results else 0
        late_success = sum(1 for r in late_results if r.success) / len(late_results) if late_results else 0
        
        return {
            'average_generation_time_ms': float(avg_generation_time),
            'early_success_rate': early_success,
            'late_success_rate': late_success,
            'learning_improvement': late_success - early_success if early_results and late_results else None,
            'suggestions_per_iteration': len(self.all_results) / max(1, self.iteration)
        }
    
    def _save_results(self, analysis: Dict):
        """Save evolution results to JSON file"""
        
        output_data = {
            'ai_evolution_results': analysis,
            'detailed_results': [
                {
                    'config': asdict(r.config),
                    'success': r.success,
                    'performance_score': r.performance_score,
                    'efficiency_ratio': r.efficiency_ratio,
                    'source': r.suggestion_source,
                    'error_message': r.validation_result.error_message,
                    'parameter_count': r.validation_result.parameter_count,
                    'memory_mb': r.validation_result.memory_requirements_mb
                }
                for r in self.all_results
            ]
        }
        
        filename = f"ai_evolution_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"💾 Results saved to {filename}")

    def run_manual_test(self, manual_configs: List[ArchitectureConfig]) -> Dict[str, Any]:
        """Test manual architectures without LLM (useful for debugging/fallback)"""
        
        logger.info(f"🔧 Testing {len(manual_configs)} manual architectures")
        start_time = time.time()
        
        # Test each manual architecture
        for config in manual_configs:
            result = self._test_architecture(config, "manual")
            self.all_results.append(result)
            
            if result.success:
                self.successful_results.append(result)
                logger.info(f"✅ Manual Success: {config.name} - {result.performance_score:.2f}")
            else:
                self.failed_results.append(result)
                logger.info(f"❌ Manual Failed: {config.name}")
        
        # Generate analysis
        total_time = time.time() - start_time
        analysis = self._analyze_evolution_results(total_time)
        
        # Save results
        self._save_results(analysis)
        
        logger.info("🏁 Manual testing complete!")
        return analysis
    
    def create_test_architectures(self) -> List[ArchitectureConfig]:
        """Create a set of test architectures for manual testing"""
        
        test_configs = [
            ArchitectureConfig(
                name="manual_small",
                hidden_dim=256,
                num_layers=4,
                num_heads=4,
                vocab_size=16000,
                max_seq_len=512,
                activation="gelu",
                normalization="rms_norm"
            ),
            ArchitectureConfig(
                name="manual_medium",
                hidden_dim=512,
                num_layers=6,
                num_heads=8,
                vocab_size=32000,
                max_seq_len=1024,
                activation="silu",
                normalization="layer_norm"
            ),
            ArchitectureConfig(
                name="manual_large",
                hidden_dim=768,
                num_layers=8,
                num_heads=12,
                vocab_size=50000,
                max_seq_len=1024,
                activation="gelu",
                normalization="rms_norm"
            ),
            # Test edge case: very small
            ArchitectureConfig(
                name="manual_tiny",
                hidden_dim=128,
                num_layers=2,
                num_heads=2,
                vocab_size=8000,
                max_seq_len=256,
                activation="relu",
                normalization="layer_norm"
            ),
            # Test efficiency focus
            ArchitectureConfig(
                name="manual_efficient",
                hidden_dim=384,
                num_layers=6,
                num_heads=6,
                vocab_size=32000,
                max_seq_len=1024,
                activation="silu",
                normalization="rms_norm"
            )
        ]
        
        logger.info(f"Created {len(test_configs)} test architectures")
        return test_configs

def main():
    """Main execution function"""
    print("🤖 AI Evolution Search - Local Phi-3 Based Architecture Evolution")
    print("=" * 80)
    
    try:
        # Check if mlx-lm is available
        try:
            import mlx_lm
            print("✅ mlx-lm available")
        except ImportError:
            print("❌ mlx-lm not found. Install with: pip install mlx-lm")
            print("   This is required for local Phi-3 support")
            return 1
        
        # Configure evolution
        config = AIEvolutionConfig(
            max_iterations=10,
            suggestions_per_iteration=4,
            temperature=0.3  # Balance creativity vs consistency
        )
        
        print(f"🔧 Configuration:")
        print(f"   • Max iterations: {config.max_iterations}")
        print(f"   • Suggestions per iteration: {config.suggestions_per_iteration}")
        print(f"   • Model: {config.model_path}")
        print(f"   • Temperature: {config.temperature}")
        
        # Run AI evolution
        evolution = AIArchitectureEvolution(config)
        results = evolution.run_evolution()
        
        # Display key results
        metadata = results['evolution_metadata']
        print(f"\n🏆 Evolution Results:")
        print(f"   • Total architectures tested: {metadata['total_architectures_tested']}")
        print(f"   • Successful: {metadata['successful_architectures']}")
        print(f"   • Success rate: {metadata['overall_success_rate']:.1%}")
        print(f"   • Total time: {metadata['total_time_seconds']:.1f}s")
        
        if results['best_architectures'] and results['best_architectures']['highest_performance']:
            best = results['best_architectures']['highest_performance']
            config_info = best['config']
            print(f"\n🥇 Best Architecture:")
            print(f"   • {config_info['num_layers']}L-{config_info['hidden_dim']}H-{config_info['num_heads']}A")
            print(f"   • Performance score: {best['performance_score']:.2f}")
            print(f"   • Parameters: {best['parameter_count']/1e6:.1f}M")
            print(f"   • Memory: {best['memory_mb']:.1f}MB")
        
        ai_perf = results['ai_performance_metrics']
        if ai_perf.get('learning_improvement') is not None:
            improvement = ai_perf['learning_improvement']
            print(f"\n🧠 AI Learning:")
            print(f"   • Success rate improvement: {improvement:+.1%}")
            print(f"   • Average generation time: {ai_perf['average_generation_time_ms']:.1f}ms")
        
        print(f"\n✅ AI Evolution complete! Check the saved results file.")
        
        return 0
        
    except Exception as e:
        logger.error(f"AI Evolution failed: {e}")
        print(f"❌ Evolution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 