# Fallback LLM System - Comprehensive Documentation

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [API Keys Setup](#api-keys-setup)
- [Standalone Program Usage](#standalone-program-usage)
- [Command-Line Flags](#command-line-flags)
- [Module Import and Usage](#module-import-and-usage)
- [Configuration Classes](#configuration-classes)
- [Examples](#examples)
- [Error Handling](#error-handling)
- [Performance and Monitoring](#performance-and-monitoring)
- [Troubleshooting](#troubleshooting)

## Overview

The Fallback LLM System is a robust, production-ready multi-provider Large Language Model client that provides intelligent fallback capabilities across multiple AI providers. It features:

- **Multi-tier fallback architecture**: Cerebras → Groq → OpenRouter
- **Model rotation within providers**: Multiple models per provider for maximum availability
- **Exponential backoff with jitter**: Smart retry logic for rate limiting
- **Dual interface**: Both command-line tool and Python module
- **Comprehensive monitoring**: Statistics tracking and detailed logging
- **Enterprise-grade reliability**: Graceful error handling and recovery

## Dependencies

### Required Python Packages
```bash
requests>=2.25.0
```

### Python Version
- **Minimum**: Python 3.7+
- **Recommended**: Python 3.8+

### System Requirements
- Internet connection for API access
- Valid API keys for at least one provider

## Installation

### Method 1: Direct Download
```bash
# Download the script
wget https://your-repo/fallback_llm.py
# or
curl -O https://your-repo/fallback_llm.py

# Install dependencies
pip install requests
```

### Method 2: Clone Repository
```bash
git clone https://your-repo/fallback-llm.git
cd fallback-llm
pip install -r requirements.txt
```

### Method 3: Pip Install (if packaged)
```bash
pip install fallback-llm
```

### Verify Installation
```bash
python fallback_llm.py --help
```

## API Keys Setup

### Environment Variables Method
```bash
export CEREBRAS_API_KEY="your_cerebras_api_key_here"
export GROQ_API_KEY="your_groq_api_key_here"
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### .env File Method (Recommended)
Create a `.env` file in the same directory as `fallback_llm.py`:
```bash
# .env file
CEREBRAS_API_KEY=csk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Shell Configuration (Persistent)
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export CEREBRAS_API_KEY="your_key"' >> ~/.bashrc
echo 'export GROQ_API_KEY="your_key"' >> ~/.bashrc
echo 'export OPENROUTER_API_KEY="your_key"' >> ~/.bashrc
source ~/.bashrc
```

### API Key Sources
- **Cerebras**: https://inference.cerebras.ai/
- **Groq**: https://console.groq.com/
- **OpenRouter**: https://openrouter.ai/

## Standalone Program Usage

### Basic Syntax
```bash
python fallback_llm.py "your question here" [options]
```

### Simple Examples
```bash
# Basic question
python fallback_llm.py "What is machine learning?"

# Question with quotes
python fallback_llm.py "Explain the concept of 'artificial intelligence'"

# Multi-line question
python fallback_llm.py "What is quantum computing?
How does it differ from classical computing?"
```

### Advanced Examples
```bash
# Limit response length
python fallback_llm.py "Explain neural networks" --max-tokens 300

# Use specific provider
python fallback_llm.py "Write a Python function" --provider groq

# Verbose output with statistics
python fallback_llm.py "Complex question" --verbose --stats

# Custom retry configuration
python fallback_llm.py "Question" --max-attempts 3 --base-delay 1.0

# Disable jitter for predictable timing
python fallback_llm.py "Question" --disable-jitter
```

## Command-Line Flags

### Required Arguments
| Argument | Type | Description |
|----------|------|-------------|
| `question` | string | The question to ask the LLM (required) |

### Optional Arguments

#### Response Configuration
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-tokens` | integer | 500 | Maximum tokens in response |

#### Provider Selection
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--provider` | choice | None | Specific provider to use |
| | | | Choices: `cerebras`, `groq`, `openrouter` |

#### Retry Configuration
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-attempts` | integer | 5 | Maximum retry attempts per model |
| `--base-delay` | float | 2.0 | Base delay for exponential backoff (seconds) |
| `--max-delay` | float | 60.0 | Maximum delay for backoff (seconds) |
| `--disable-jitter` | flag | False | Disable jitter in exponential backoff |

#### Output and Debugging
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--verbose`, `-v` | flag | False | Enable verbose output |
| `--stats` | flag | False | Show usage statistics after response |

### Flag Examples
```bash
# Response configuration
python fallback_llm.py "Question" --max-tokens 1000

# Provider selection
python fallback_llm.py "Question" --provider cerebras
python fallback_llm.py "Question" --provider groq
python fallback_llm.py "Question" --provider openrouter

# Retry configuration
python fallback_llm.py "Question" --max-attempts 3
python fallback_llm.py "Question" --base-delay 1.5
python fallback_llm.py "Question" --max-delay 30.0
python fallback_llm.py "Question" --disable-jitter

# Output options
python fallback_llm.py "Question" --verbose
python fallback_llm.py "Question" --stats
python fallback_llm.py "Question" -v --stats
```

## Module Import and Usage

### Basic Import
```python
from fallback_llm import FallbackLLM
```

### Import with Configuration
```python
from fallback_llm import FallbackLLM, RetryConfig, ProviderStatus, ProviderHealth
```

### Basic Module Usage
```python
# Initialize with defaults
llm = FallbackLLM()

# Ask a question
response = llm.ask("What is artificial intelligence?")
print(response)
```

### Advanced Module Usage
```python
# Custom retry configuration
from fallback_llm import FallbackLLM, RetryConfig

config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

# Initialize with custom config
llm = FallbackLLM(retry_config=config)

# Ask with parameters
response = llm.ask(
    question="Explain machine learning algorithms",
    max_tokens=500,
    provider="groq"  # Optional: specific provider
)
```

### Module Methods
```python
class FallbackLLM:
    def __init__(self, retry_config: Optional[RetryConfig] = None)
    def ask(self, question: str, max_tokens: int = 500, provider: Optional[str] = None) -> str
    def get_stats(self) -> Dict
    def reset_stats(self) -> None
```

## Configuration Classes

### RetryConfig Class
```python
@dataclass
class RetryConfig:
    max_attempts: int = 5        # Maximum retry attempts per model
    base_delay: float = 2.0      # Base delay for exponential backoff
    max_delay: float = 60.0      # Maximum delay cap
    exponential_base: float = 2.0 # Exponential base (2.0 = doubling)
    jitter: bool = True          # Enable jitter (80-100% of delay)
```

### ProviderStatus Enum
```python
class ProviderStatus(Enum):
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
```

### ProviderHealth Class
```python
@dataclass
class ProviderHealth:
    status: ProviderStatus = ProviderStatus.HEALTHY
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0
    rate_limit_reset_time: float = 0.0
```

## Examples

### Example 1: Basic CLI Usage
```bash
$ python fallback_llm.py "What is the capital of France?"

🔄 Trying provider: cerebras
🎯 Trying cerebras model 1/4: llama-4-scout-17b-16e-instruct
✅ Success with cerebras llama-4-scout-17b-16e-instruct on attempt 1

💡 Response:
The capital of France is Paris.

⏱️  Response time: 1.01s
```

### Example 2: Verbose Output with Statistics
```bash
$ python fallback_llm.py "Explain machine learning" --verbose --stats --max-tokens 200

🚀 Initializing Fallback LLM System...
✅ Available providers: cerebras, groq, openrouter
❓ Question: Explain machine learning
🤔 Thinking...
🔄 Trying provider: cerebras
🎯 Trying cerebras model 1/4: llama-4-scout-17b-16e-instruct
✅ Success with cerebras llama-4-scout-17b-16e-instruct on attempt 1

💡 Response:
Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions based on that data.

⏱️  Response time: 1.45s

📊 Usage Statistics:
   total_requests: 1
   successful_requests: 1
   failed_requests: 0
   rate_limited_requests: 0
   provider_switches: 0
```

### Example 3: Rate Limiting with Fallback
```bash
$ python fallback_llm.py "Complex analysis question" --verbose

🔄 Trying provider: cerebras
🎯 Trying cerebras model 1/4: llama-4-scout-17b-16e-instruct
⏳ Waiting 1.89s before retry...
⏳ Waiting 3.76s before retry...
⏳ Waiting 7.23s before retry...
🔄 cerebras llama-4-scout-17b-16e-instruct exhausted all 5 retry attempts
🔄 cerebras llama-4-scout-17b-16e-instruct failed, trying next model...
🎯 Trying cerebras model 2/4: llama-3.3-70b
✅ Success with cerebras llama-3.3-70b on attempt 1

💡 Response:
[Response content here]
```

### Example 4: Basic Module Usage
```python
#!/usr/bin/env python3
from fallback_llm import FallbackLLM

def main():
    # Initialize LLM
    llm = FallbackLLM()
    
    # Ask a question
    question = "What are the benefits of renewable energy?"
    response = llm.ask(question, max_tokens=300)
    
    print(f"Question: {question}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
```

### Example 5: Custom Configuration Module Usage
```python
#!/usr/bin/env python3
from fallback_llm import FallbackLLM, RetryConfig

def main():
    # Custom retry configuration
    config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=20.0,
        jitter=True
    )
    
    # Initialize with custom config
    llm = FallbackLLM(retry_config=config)
    
    # Multiple questions
    questions = [
        "What is Python programming?",
        "Explain object-oriented programming",
        "What are design patterns?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        response = llm.ask(question, max_tokens=200)
        print(f"Response {i}: {response}")
    
    # Show statistics
    stats = llm.get_stats()
    print(f"\nSession Statistics:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Successful requests: {stats['successful_requests']}")
    print(f"Success rate: {stats['successful_requests']/stats['total_requests']:.1%}")

if __name__ == "__main__":
    main()
```

### Example 6: Error Handling
```python
#!/usr/bin/env python3
from fallback_llm import FallbackLLM
import sys

def main():
    try:
        llm = FallbackLLM()
        
        # Check if any providers are available
        available_providers = [p for p in llm.fallback_order if llm.api_keys.get(p)]
        if not available_providers:
            print("❌ No API keys found!")
            print("Please set environment variables: CEREBRAS_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY")
            sys.exit(1)
        
        print(f"✅ Available providers: {', '.join(available_providers)}")
        
        # Ask question
        question = "What is quantum computing?"
        response = llm.ask(question)
        
        if response:
            print(f"Response: {response}")
        else:
            print("❌ No response received from any provider")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Example 7: Batch Processing
```python
#!/usr/bin/env python3
from fallback_llm import FallbackLLM
import time

def batch_process_questions(questions, max_tokens=300):
    """Process multiple questions in batch"""
    llm = FallbackLLM()
    results = []
    
    start_time = time.time()
    
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{len(questions)}: {question[:50]}...")
        
        response = llm.ask(question, max_tokens=max_tokens)
        results.append({
            'question': question,
            'response': response,
            'success': bool(response)
        })
    
    end_time = time.time()
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nBatch Processing Summary:")
    print(f"Total questions: {len(questions)}")
    print(f"Successful responses: {successful}")
    print(f"Success rate: {successful/len(questions):.1%}")
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Average time per question: {(end_time - start_time)/len(questions):.2f}s")
    
    return results

def main():
    questions = [
        "What is artificial intelligence?",
        "Explain machine learning algorithms",
        "What is deep learning?",
        "How do neural networks work?",
        "What is natural language processing?"
    ]
    
    results = batch_process_questions(questions)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {result['question']}")
        print(f"A: {result['response'][:100]}..." if result['response'] else "No response")

if __name__ == "__main__":
    main()
```

### Example 8: Interactive Chat
```python
#!/usr/bin/env python3
from fallback_llm import FallbackLLM

def interactive_chat():
    """Interactive chat session"""
    llm = FallbackLLM()
    
    print("🤖 Fallback LLM Chat")
    print("Type 'quit', 'exit', or 'bye' to end the session")
    print("Type 'stats' to see usage statistics")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            question = input("\n👤 You: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            # Check for stats command
            if question.lower() == 'stats':
                stats = llm.get_stats()
                print("📊 Session Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            
            # Skip empty input
            if not question:
                continue
            
            # Get response
            print("🤖 AI: ", end="", flush=True)
            response = llm.ask(question, max_tokens=400)
            
            if response:
                print(response)
            else:
                print("Sorry, I couldn't generate a response.")
                
        except KeyboardInterrupt:
            print("\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    interactive_chat()
```

## Error Handling

### Common Errors and Solutions

#### 1. Missing API Keys
```bash
❌ No API keys found! Please set environment variables:
   CEREBRAS_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY
```
**Solution**: Set up API keys as described in [API Keys Setup](#api-keys-setup)

#### 2. Network Connection Issues
```bash
❌ Error in cerebras API request: HTTPSConnectionPool(...): Max retries exceeded
```
**Solution**: Check internet connection and firewall settings

#### 3. Invalid API Key
```bash
❌ Error in cerebras API request: 401 Client Error: Unauthorized
```
**Solution**: Verify API key is correct and active

#### 4. Rate Limiting
```bash
⏳ Waiting 3.76s before retry...
```
**Solution**: This is normal behavior - the system handles rate limiting automatically

#### 5. All Providers Failed
```bash
❌ All providers and models failed
```
**Solution**: Check API keys, network connection, and provider status

### Error Handling in Code
```python
from fallback_llm import FallbackLLM

def safe_ask(question, max_retries=3):
    """Safely ask a question with error handling"""
    llm = FallbackLLM()
    
    for attempt in range(max_retries):
        try:
            response = llm.ask(question)
            if response:
                return response
            else:
                print(f"Attempt {attempt + 1}: No response received")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error - {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return None
```

## Performance and Monitoring

### Response Times by Provider
- **Cerebras**: 1-2 seconds (fastest)
- **Groq**: 1-3 seconds (very fast)
- **OpenRouter**: 2-5 seconds (variable)

### Rate Limiting Behavior
- **Exponential Backoff**: 2s → 4s → 8s → 16s → 32s
- **Jitter**: 80-100% of calculated delay
- **Model Rotation**: Tries next model after exhaustion
- **Provider Escalation**: Switches providers after all models fail

### Statistics Tracking
```python
stats = llm.get_stats()
# Returns:
{
    'total_requests': 10,
    'successful_requests': 9,
    'failed_requests': 1,
    'rate_limited_requests': 3,
    'provider_switches': 1
}
```

### Performance Monitoring
```python
import time

start_time = time.time()
response = llm.ask("Your question")
end_time = time.time()

print(f"Response time: {end_time - start_time:.2f}s")
print(f"Response length: {len(response)} characters")
```

## Troubleshooting

### Debug Mode
```bash
python fallback_llm.py "Question" --verbose
```

### Check API Keys
```python
from fallback_llm import FallbackLLM
llm = FallbackLLM()
print("Available providers:", [p for p in llm.fallback_order if llm.api_keys.get(p)])
```

### Test Individual Providers
```bash
python fallback_llm.py "Test question" --provider cerebras
python fallback_llm.py "Test question" --provider groq
python fallback_llm.py "Test question" --provider openrouter
```

### Common Issues

1. **Slow responses**: Normal during rate limiting
2. **Empty responses**: Check API keys and provider status
3. **Connection errors**: Check internet and firewall
4. **Import errors**: Ensure `requests` package is installed

### Getting Help
- Check the verbose output: `--verbose`
- Review statistics: `--stats`
- Test with simple questions first
- Verify API keys are valid and active

---

**The Fallback LLM System provides enterprise-grade reliability with comprehensive error handling and monitoring capabilities!** 🚀🤖📊
