#!/usr/bin/env python3
"""
Test script for Fallback LLM System
Verifies that the system is properly set up and working
"""

import sys
import os
import time

def test_imports():
    """Test that the module can be imported"""
    print("🔍 Testing imports...")
    try:
        from fallback_llm import FallbackLLM, RetryConfig
        print("✅ Imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_api_keys():
    """Test that API keys are available"""
    print("\n🔑 Testing API keys...")
    try:
        from fallback_llm import FallbackLLM
        llm = FallbackLLM()
        
        available_providers = [p for p in llm.fallback_order if llm.api_keys.get(p)]
        
        if available_providers:
            print(f"✅ Available providers: {', '.join(available_providers)}")
            return True
        else:
            print("❌ No API keys found")
            print("   Please set up your .env file with valid API keys")
            return False
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    try:
        from fallback_llm import FallbackLLM
        llm = FallbackLLM()
        
        # Simple test question
        test_question = "What is 2+2?"
        print(f"   Question: {test_question}")
        
        start_time = time.time()
        response = llm.ask(test_question, max_tokens=50)
        end_time = time.time()
        
        if response:
            print(f"✅ Response received: {response[:100]}...")
            print(f"   Response time: {end_time - start_time:.2f}s")
            return True
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_configuration():
    """Test custom configuration"""
    print("\n⚙️  Testing custom configuration...")
    try:
        from fallback_llm import FallbackLLM, RetryConfig
        
        # Custom config
        config = RetryConfig(
            max_attempts=2,
            base_delay=0.5,
            jitter=True
        )
        
        llm = FallbackLLM(retry_config=config)
        
        # Test with custom config
        response = llm.ask("What is AI?", max_tokens=30)
        
        if response:
            print("✅ Custom configuration working")
            return True
        else:
            print("❌ Custom configuration failed")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_statistics():
    """Test statistics tracking"""
    print("\n📊 Testing statistics...")
    try:
        from fallback_llm import FallbackLLM
        llm = FallbackLLM()
        
        # Reset stats
        llm.reset_stats()
        
        # Make a request
        llm.ask("Test question", max_tokens=20)
        
        # Check stats
        stats = llm.get_stats()
        
        if stats['total_requests'] > 0:
            print("✅ Statistics tracking working")
            print(f"   Total requests: {stats['total_requests']}")
            print(f"   Successful requests: {stats['successful_requests']}")
            return True
        else:
            print("❌ Statistics not updating")
            return False
            
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False

def test_cli_help():
    """Test CLI help functionality"""
    print("\n📋 Testing CLI help...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "fallback_llm.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "usage:" in result.stdout.lower():
            print("✅ CLI help working")
            return True
        else:
            print("❌ CLI help failed")
            return False
            
    except Exception as e:
        print(f"❌ CLI help test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Fallback LLM System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("API Keys", test_api_keys),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration", test_configuration),
        ("Statistics", test_statistics),
        ("CLI Help", test_cli_help)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
