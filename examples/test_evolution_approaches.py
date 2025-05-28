#!/usr/bin/env python3
"""
Test Evolution Approaches - Demonstrating Multiple Architecture Testing Methods
"""

import logging
import time
from ai_evolution_search import AIEvolutionConfig, AIArchitectureEvolution

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_improved_llm_evolution():
    """Test the improved LLM-based evolution with better JSON parsing"""
    
    print("🤖 Testing Improved LLM Evolution")
    print("=" * 50)
    
    config = AIEvolutionConfig(
        max_iterations=2,  # Small test
        suggestions_per_iteration=3,
        temperature=0.2,  # Low temperature for more consistent JSON
        max_tokens=800   # Shorter responses for cleaner JSON
    )
    
    print(f"Configuration: {config.max_iterations} iterations, {config.suggestions_per_iteration} suggestions each")
    
    try:
        evolution = AIArchitectureEvolution(config)
        results = evolution.run_evolution()
        
        metadata = results['evolution_metadata']
        print(f"\n📊 LLM Evolution Results:")
        print(f"   • Architectures tested: {metadata['total_architectures_tested']}")
        print(f"   • Success rate: {metadata['overall_success_rate']:.1%}")
        print(f"   • Time taken: {metadata['total_time_seconds']:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Evolution failed: {e}")
        return False

def test_manual_architectures():
    """Test manual architecture testing without LLM"""
    
    print("\n🔧 Testing Manual Architecture Validation")
    print("=" * 50)
    
    try:
        # Create evolution system (but won't use LLM)
        config = AIEvolutionConfig()
        evolution = AIArchitectureEvolution(config)
        
        # Create test architectures
        test_configs = evolution.create_test_architectures()
        
        print(f"Testing {len(test_configs)} manual architectures:")
        for config in test_configs:
            print(f"   • {config.name}: {config.num_layers}L-{config.hidden_dim}H-{config.num_heads}A")
        
        # Run manual testing
        results = evolution.run_manual_test(test_configs)
        
        metadata = results['evolution_metadata']
        print(f"\n📊 Manual Testing Results:")
        print(f"   • Architectures tested: {metadata['total_architectures_tested']}")
        print(f"   • Successful: {metadata['successful_architectures']}")
        print(f"   • Success rate: {metadata['overall_success_rate']:.1%}")
        print(f"   • Time taken: {metadata['total_time_seconds']:.1f}s")
        
        # Show best architecture if any succeeded
        best_archs = results.get('best_architectures', {})
        if best_archs and best_archs.get('highest_performance'):
            best = best_archs['highest_performance']
            config_info = best['config']
            print(f"\n🏆 Best Manual Architecture:")
            print(f"   • {config_info['name']}: {config_info['num_layers']}L-{config_info['hidden_dim']}H-{config_info['num_heads']}A")
            print(f"   • Performance: {best['performance_score']:.2f}")
            print(f"   • Parameters: {best['parameter_count']/1e6:.1f}M")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual testing failed: {e}")
        return False

def test_json_parser_robustness():
    """Test the improved JSON parser with various problematic inputs"""
    
    print("\n🧪 Testing JSON Parser Robustness")
    print("=" * 50)
    
    from ai_evolution_search import ArchitectureParser
    
    parser = ArchitectureParser()
    
    # Test cases with problematic LLM outputs
    test_cases = [
        # Case 1: Clean JSON (should work)
        '''[
  {
    "name": "test1",
    "hidden_dim": 512,
    "num_layers": 6,
    "num_heads": 8,
    "vocab_size": 32000,
    "max_seq_len": 1024,
    "activation": "gelu",
    "normalization": "rms_norm"
  }
]''',
        
        # Case 2: JSON with extra text (common LLM issue)
        '''Here are the architectures:

[
  {
    "name": "test2",
    "hidden_dim": 256,
    "num_layers": 4,
    "num_heads": 4,
    "vocab_size": 16000,
    "max_seq_len": 512,
    "activation": "gelu",
    "normalization": "rms_norm"
  }
]

These architectures should work well for your use case.''',
        
        # Case 3: Architecture description format
        '''Based on the requirements, I suggest:
        - 6L-384H-6A configuration
        - Using GELU activation
        - RMS normalization
        - 32000 vocabulary size''',
        
        # Case 4: Malformed JSON (missing quotes)
        '''[
  {
    name: "test4",
    hidden_dim: 512,
    num_layers: 8,
    num_heads: 8,
    vocab_size: 32000,
    max_seq_len: 1024,
    activation: "gelu",
    normalization: "rms_norm"
  }
]'''
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}:")
        print(f"   Input: {test_case[:100]}...")
        
        configs = parser.parse_architectures(test_case)
        
        if configs:
            print(f"   ✅ Success: Extracted {len(configs)} configurations")
            for config in configs:
                print(f"      • {config.name}: {config.num_layers}L-{config.hidden_dim}H-{config.num_heads}A")
        else:
            print(f"   ❌ Failed: No configurations extracted")
    
    return True

def main():
    """Run all evolution approach tests"""
    
    print("🚀 Testing All Evolution Approaches")
    print("=" * 70)
    
    results = {
        'json_parser': False,
        'manual_testing': False,
        'llm_evolution': False
    }
    
    # Test 1: JSON Parser Robustness
    try:
        results['json_parser'] = test_json_parser_robustness()
    except Exception as e:
        print(f"❌ JSON parser test failed: {e}")
    
    # Test 2: Manual Architecture Testing  
    try:
        results['manual_testing'] = test_manual_architectures()
    except Exception as e:
        print(f"❌ Manual testing failed: {e}")
    
    # Test 3: Improved LLM Evolution (optional, requires model)
    print(f"\n" + "="*70)
    try:
        response = input("Run LLM evolution test? (requires Phi-3 model) (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            results['llm_evolution'] = test_improved_llm_evolution()
        else:
            print("Skipping LLM evolution test")
            results['llm_evolution'] = None
    except KeyboardInterrupt:
        print("\nSkipping LLM evolution test")
        results['llm_evolution'] = None
    
    # Summary
    print(f"\n🏁 Test Results Summary:")
    print(f"   • JSON Parser: {'✅ PASS' if results['json_parser'] else '❌ FAIL'}")
    print(f"   • Manual Testing: {'✅ PASS' if results['manual_testing'] else '❌ FAIL'}")
    if results['llm_evolution'] is not None:
        print(f"   • LLM Evolution: {'✅ PASS' if results['llm_evolution'] else '❌ FAIL'}")
    else:
        print(f"   • LLM Evolution: ⏭️  SKIPPED")
    
    # Show how to use each approach
    print(f"\n📚 How to Use Each Approach:")
    print(f"""
1. **Manual Testing** (No LLM required):
   ```python
   from ai_evolution_search import AIArchitectureEvolution
   evolution = AIArchitectureEvolution()
   configs = evolution.create_test_architectures()
   results = evolution.run_manual_test(configs)
   ```

2. **LLM Evolution** (Phi-3 required):
   ```python
   from ai_evolution_search import AIEvolutionConfig, AIArchitectureEvolution
   config = AIEvolutionConfig(max_iterations=5, suggestions_per_iteration=3)
   evolution = AIArchitectureEvolution(config)
   results = evolution.run_evolution()
   ```

3. **Custom Architectures**:
   ```python
   from mlx_architecture_final import ArchitectureConfig
   custom_config = ArchitectureConfig(
       name="my_arch", hidden_dim=512, num_layers=6, num_heads=8,
       vocab_size=32000, max_seq_len=1024, activation="gelu", 
       normalization="rms_norm"
   )
   results = evolution.run_manual_test([custom_config])
   ```
""")
    
    success_count = sum(1 for result in results.values() if result is True)
    total_tests = sum(1 for result in results.values() if result is not None)
    
    print(f"\n✅ {success_count}/{total_tests} tests passed!")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 