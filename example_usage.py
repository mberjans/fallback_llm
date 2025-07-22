#!/usr/bin/env python3
"""
Example usage of the Fallback LLM System
Demonstrates both module import and various use cases
"""

from fallback_llm import FallbackLLM, RetryConfig
import time

def example_basic_usage():
    """Basic usage example"""
    print("🔹 Basic Usage Example")
    print("-" * 30)
    
    # Initialize with default settings
    llm = FallbackLLM()
    
    # Ask a simple question
    question = "What is artificial intelligence in one sentence?"
    print(f"Question: {question}")
    
    response = llm.ask(question)
    print(f"Response: {response}")
    print()

def example_custom_config():
    """Custom configuration example"""
    print("🔹 Custom Configuration Example")
    print("-" * 35)
    
    # Create custom retry configuration
    custom_config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        jitter=True
    )
    
    # Initialize with custom config
    llm = FallbackLLM(retry_config=custom_config)
    
    # Ask with custom max_tokens
    question = "Explain machine learning algorithms briefly"
    print(f"Question: {question}")
    
    response = llm.ask(question, max_tokens=300)
    print(f"Response: {response}")
    print()

def example_specific_provider():
    """Specific provider example"""
    print("🔹 Specific Provider Example")
    print("-" * 32)
    
    llm = FallbackLLM()
    
    # Try specific provider
    question = "What is quantum computing?"
    print(f"Question: {question}")
    print("Using Groq provider specifically...")
    
    response = llm.ask(question, provider="groq")
    print(f"Response: {response}")
    print()

def example_multiple_questions():
    """Multiple questions example"""
    print("🔹 Multiple Questions Example")
    print("-" * 32)
    
    llm = FallbackLLM()
    
    questions = [
        "What is Python programming?",
        "Explain blockchain technology",
        "What are neural networks?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        response = llm.ask(question, max_tokens=200)
        print(f"Response {i}: {response[:100]}...")
        print()

def example_with_statistics():
    """Statistics tracking example"""
    print("🔹 Statistics Tracking Example")
    print("-" * 33)
    
    llm = FallbackLLM()
    
    # Reset stats
    llm.reset_stats()
    
    # Ask several questions
    questions = [
        "What is data science?",
        "Explain cloud computing",
        "What is DevOps?"
    ]
    
    start_time = time.time()
    
    for question in questions:
        response = llm.ask(question, max_tokens=150)
        print(f"Q: {question}")
        print(f"A: {response[:80]}...")
        print()
    
    end_time = time.time()
    
    # Show statistics
    stats = llm.get_stats()
    print("📊 Session Statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Successful requests: {stats['successful_requests']}")
    print(f"   Failed requests: {stats['failed_requests']}")
    print(f"   Rate limited requests: {stats['rate_limited_requests']}")
    print(f"   Total time: {end_time - start_time:.2f}s")
    print()

def example_error_handling():
    """Error handling example"""
    print("🔹 Error Handling Example")
    print("-" * 28)
    
    llm = FallbackLLM()
    
    # Try with invalid provider
    question = "What is software engineering?"
    print(f"Question: {question}")
    print("Trying with invalid provider...")
    
    response = llm.ask(question, provider="invalid_provider")
    if response:
        print(f"Response: {response}")
    else:
        print("No response (as expected with invalid provider)")
    
    # Try with fallback system
    print("\nTrying with fallback system...")
    response = llm.ask(question)
    print(f"Response: {response[:100]}...")
    print()

def example_conversation():
    """Conversation-like example"""
    print("🔹 Conversation Example")
    print("-" * 24)
    
    llm = FallbackLLM()
    
    # Simulate a conversation
    conversation = [
        "What is machine learning?",
        "Can you give me an example of supervised learning?",
        "What about unsupervised learning?",
        "How is deep learning different?"
    ]
    
    for i, question in enumerate(conversation, 1):
        print(f"Turn {i}: {question}")
        response = llm.ask(question, max_tokens=250)
        print(f"AI: {response}")
        print("-" * 50)

def main():
    """Run all examples"""
    print("🚀 Fallback LLM System - Usage Examples")
    print("=" * 50)
    print()
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_config()
        example_specific_provider()
        example_multiple_questions()
        example_with_statistics()
        example_error_handling()
        example_conversation()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure you have API keys set in environment variables or .env file")

if __name__ == "__main__":
    main()
