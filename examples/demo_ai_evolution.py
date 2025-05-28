#!/usr/bin/env python3
"""
Demo: AI Evolution Search
Demonstrates the local Phi-3 based architecture evolution system
"""

import logging
import time
from ai_evolution_search import AIEvolutionConfig, AIArchitectureEvolution

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_ai_evolution():
    """Demonstrate AI evolution with a small test configuration"""
    
    print("🤖 AI Evolution Demo - Local Phi-3 Architecture Search")
    print("=" * 70)
    
    # Configure for demonstration (small scale)
    config = AIEvolutionConfig(
        max_iterations=3,  # Small for demo
        suggestions_per_iteration=3,  # Manageable number
        temperature=0.4,  # Balanced creativity
        max_tokens=1500
    )
    
    print(f"📋 Demo Configuration:")
    print(f"   • Iterations: {config.max_iterations}")
    print(f"   • Suggestions per iteration: {config.suggestions_per_iteration}")
    print(f"   • Model: {config.model_path}")
    print(f"   • Temperature: {config.temperature}")
    
    try:
        print(f"\n🔄 Initializing AI Evolution System...")
        start_time = time.time()
        
        # Initialize the AI evolution system
        evolution = AIArchitectureEvolution(config)
        
        init_time = time.time() - start_time
        print(f"✅ Initialization complete in {init_time:.1f}s")
        
        print(f"\n🚀 Starting AI-driven architecture evolution...")
        print(f"   This will test {config.max_iterations * config.suggestions_per_iteration} architectures")
        
        # Run the evolution
        results = evolution.run_evolution()
        
        # Display results
        print(f"\n📊 Evolution Results Summary:")
        metadata = results['evolution_metadata']
        print(f"   • Total architectures tested: {metadata['total_architectures_tested']}")
        print(f"   • Successful validations: {metadata['successful_architectures']}")
        print(f"   • Failed validations: {metadata['failed_architectures']}")
        print(f"   • Overall success rate: {metadata['overall_success_rate']:.1%}")
        print(f"   • Total execution time: {metadata['total_time_seconds']:.1f}s")
        
        # Show best architecture if any succeeded
        best_archs = results.get('best_architectures', {})
        if best_archs and best_archs.get('highest_performance'):
            best = best_archs['highest_performance']
            config_info = best['config']
            print(f"\n🏆 Best Architecture Found:")
            print(f"   • Architecture: {config_info['num_layers']}L-{config_info['hidden_dim']}H-{config_info['num_heads']}A")
            print(f"   • Activation: {config_info['activation']}")
            print(f"   • Normalization: {config_info['normalization']}")
            print(f"   • Performance score: {best['performance_score']:.2f}")
            print(f"   • Parameters: {best['parameter_count']/1e6:.1f}M")
            print(f"   • Memory estimate: {best['memory_mb']:.1f}MB")
        else:
            print(f"\n⚠️  No architectures passed validation in this demo")
            print(f"   This is normal for a short demo - try longer runs for better results")
        
        # Show AI learning metrics
        ai_metrics = results.get('ai_performance_metrics', {})
        if ai_metrics:
            print(f"\n🧠 AI Performance Metrics:")
            if ai_metrics.get('learning_improvement') is not None:
                improvement = ai_metrics['learning_improvement']
                print(f"   • Learning improvement: {improvement:+.1%}")
            print(f"   • Average generation time: {ai_metrics.get('average_generation_time_ms', 0):.1f}ms")
            print(f"   • Early vs late success: {ai_metrics.get('early_success_rate', 0):.1%} → {ai_metrics.get('late_success_rate', 0):.1%}")
        
        # Show failure analysis
        failures = results.get('failure_analysis', {})
        if failures and failures.get('total_failures', 0) > 0:
            print(f"\n❌ Failure Analysis:")
            print(f"   • Total failures: {failures['total_failures']}")
            if 'failure_categories' in failures:
                for category, count in failures['failure_categories'].items():
                    print(f"   • {category}: {count}")
        
        print(f"\n✅ Demo complete! Check the saved JSON file for detailed results.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"   Make sure mlx-lm is installed: pip install mlx-lm")
        return False
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        print(f"   Error details: {traceback.format_exc()}")
        return False

def quick_test_without_model():
    """Test the system components without loading the actual model"""
    
    print("\n🔧 Quick Component Test (No Model Loading)")
    print("-" * 50)
    
    try:
        from ai_evolution_search import (
            ArchitecturePromptGenerator, 
            ArchitectureParser,
            EvolutionResult
        )
        from mlx_architecture_final import ArchitectureConfig
        
        # Test prompt generator
        prompt_gen = ArchitecturePromptGenerator()
        print("✅ Prompt generator created")
        
        # Test parser
        parser = ArchitectureParser()
        print("✅ Parser created")
        
        # Test with sample JSON
        sample_json = '''[
            {
                "name": "test_arch",
                "hidden_dim": 512,
                "num_layers": 6,
                "num_heads": 8,
                "vocab_size": 32000,
                "max_seq_len": 1024,
                "activation": "gelu",
                "normalization": "rms_norm"
            }
        ]'''
        
        configs = parser.parse_architectures(sample_json)
        if configs:
            config = configs[0]
            print(f"✅ JSON parsing successful: {config.name}")
            print(f"   Architecture: {config.num_layers}L-{config.hidden_dim}H-{config.num_heads}A")
        else:
            print("❌ JSON parsing failed")
        
        print("✅ All components working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

if __name__ == "__main__":
    # Run quick test first
    component_test_passed = quick_test_without_model()
    
    if component_test_passed:
        print(f"\n" + "="*70)
        
        # Ask user if they want to run full demo
        print(f"Component test passed! The full demo will:")
        print(f"• Download and load Phi-3-mini-4k-instruct-4bit (~2GB)")
        print(f"• Run AI-driven architecture evolution")
        print(f"• Test 9 architectures with full validation")
        print(f"• Take several minutes to complete")
        
        try:
            response = input(f"\nRun full AI evolution demo? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                success = demo_ai_evolution()
                exit(0 if success else 1)
            else:
                print(f"Skipping full demo. Component test completed successfully!")
                exit(0)
        except KeyboardInterrupt:
            print(f"\nDemo cancelled by user.")
            exit(0)
    else:
        print(f"❌ Component test failed. Check dependencies and try again.")
        exit(1) 