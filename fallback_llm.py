#!/usr/bin/env python3
"""
Fallback LLM System - Standalone Multi-Provider LLM Client
Can be used as a command-line tool or imported as a module

Features:
- Multi-tier fallback (Cerebras → Groq → OpenRouter)
- Exponential backoff with jitter
- Model rotation within providers
- Comprehensive error handling
- CLI interface and module interface

Usage as CLI:
    python fallback_llm.py "What is machine learning?"
    python fallback_llm.py "Explain quantum computing" --max-tokens 500 --provider groq

Usage as module:
    from fallback_llm import FallbackLLM
    llm = FallbackLLM()
    response = llm.ask("What is artificial intelligence?")
"""

import os
import sys
import time
import json
import random
import requests
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ProviderStatus(Enum):
    """Provider status enumeration"""
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"

@dataclass
class ProviderHealth:
    """Provider health tracking"""
    status: ProviderStatus = ProviderStatus.HEALTHY
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0
    rate_limit_reset_time: float = 0.0

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

class FallbackLLM:
    """Multi-provider LLM client with intelligent fallback"""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """Initialize the fallback LLM system"""
        self.retry_config = retry_config or RetryConfig()
        self.fallback_order = ['cerebras', 'groq', 'openrouter']
        self.provider_health = {
            provider: ProviderHealth() for provider in self.fallback_order
        }
        
        # Load API keys from environment
        self.api_keys = self._load_api_keys()
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'provider_switches': 0
        }
        
        # Model configurations
        self.models = {
            'cerebras': [
                'llama-4-scout-17b-16e-instruct',
                'llama-3.3-70b',
                'llama3.1-8b',
                'qwen-3-32b'
            ],
            'groq': [
                'meta-llama/llama-4-maverick-17b-128e-instruct',
                'meta-llama/llama-4-scout-17b-16e-instruct',
                'qwen/qwen3-32b',
                'llama-3.1-8b-instant',
                'llama-3.3-70b-versatile'
            ],
            'openrouter': [
                'mistralai/mistral-nemo:free',
                'tngtech/deepseek-r1t-chimera:free',
                'google/gemini-2.0-flash-exp:free',
                'mistralai/mistral-small-3.1-24b-instruct:free'
            ]
        }
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables"""
        api_keys = {}
        
        # Try to load from .env file first
        env_file = '.env'
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
        
        # Load API keys
        api_keys['cerebras'] = os.getenv('CEREBRAS_API_KEY')
        api_keys['groq'] = os.getenv('GROQ_API_KEY')
        api_keys['openrouter'] = os.getenv('OPENROUTER_API_KEY')
        
        return api_keys
    
    def ask(self, question: str, max_tokens: int = 500, provider: Optional[str] = None) -> str:
        """
        Ask a question using the fallback LLM system
        
        Args:
            question: The question to ask
            max_tokens: Maximum tokens in response
            provider: Specific provider to use (optional)
            
        Returns:
            The LLM response as a string
        """
        if provider and provider in self.fallback_order:
            # Use specific provider
            return self._ask_provider(provider, question, max_tokens)
        else:
            # Use multi-tier fallback
            return self._ask_with_fallback(question, max_tokens)
    
    def _ask_with_fallback(self, question: str, max_tokens: int) -> str:
        """Ask question with multi-tier fallback"""
        self.stats['total_requests'] += 1
        
        # Try each provider in fallback order
        for provider_name in self.fallback_order:
            print(f"🔄 Trying provider: {provider_name}")
            
            # Get models for this provider
            provider_models = self.models.get(provider_name, [])
            if not provider_models:
                print(f"⚠️  No models available for {provider_name}, skipping...")
                continue
            
            # Try each model in this provider
            for model_index, model_id in enumerate(provider_models):
                print(f"🎯 Trying {provider_name} model {model_index + 1}/{len(provider_models)}: {model_id}")
                
                # Try this specific model with exponential backoff
                for attempt in range(self.retry_config.max_attempts):
                    response, success, is_rate_limit = self._make_api_request(
                        provider_name, model_id, question, max_tokens
                    )
                    
                    if success:
                        self.stats['successful_requests'] += 1
                        print(f"✅ Success with {provider_name} {model_id} on attempt {attempt + 1}")
                        return response
                    
                    # Handle rate limiting with exponential backoff
                    if is_rate_limit:
                        self.stats['rate_limited_requests'] += 1
                        
                        if attempt < self.retry_config.max_attempts - 1:
                            delay = self._calculate_delay(attempt, provider_name, model_id)
                            print(f"⏳ Waiting {delay:.2f}s before retry...")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"🔄 {provider_name} {model_id} exhausted all {self.retry_config.max_attempts} retry attempts")
                            break
                    else:
                        # Non-rate-limit error, try next model
                        print(f"❌ {provider_name} {model_id} API error on attempt {attempt + 1}, trying next model...")
                        break
                
                # Try next model in same provider
                print(f"🔄 {provider_name} {model_id} failed, trying next model...")
            
            # Try next provider after all models exhausted
            print(f"🚀 All {provider_name} models exhausted, escalating to next provider...")
        
        # All providers failed
        self.stats['failed_requests'] += 1
        print(f"❌ All providers and models failed")
        return ""
    
    def _ask_provider(self, provider: str, question: str, max_tokens: int) -> str:
        """Ask question using specific provider"""
        models = self.models.get(provider, [])
        if not models:
            return ""
        
        model_id = models[0]  # Use first model
        response, success, _ = self._make_api_request(provider, model_id, question, max_tokens)
        return response if success else ""
    
    def _make_api_request(self, provider: str, model_id: str, question: str, max_tokens: int) -> Tuple[str, bool, bool]:
        """Make API request to specific provider with specific model"""
        try:
            if provider == "cerebras":
                return self._cerebras_request(model_id, question, max_tokens)
            elif provider == "groq":
                return self._groq_request(model_id, question, max_tokens)
            elif provider == "openrouter":
                return self._openrouter_request(model_id, question, max_tokens)
            else:
                return "", False, False
        except Exception as e:
            print(f"❌ Error in {provider} API request: {e}")
            return "", False, False
    
    def _cerebras_request(self, model_id: str, question: str, max_tokens: int) -> Tuple[str, bool, bool]:
        """Make request to Cerebras API"""
        api_key = self.api_keys.get('cerebras')
        if not api_key:
            return "", False, False
        
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 429:
            return "", False, True  # Rate limited
        
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip(), True, False
    
    def _groq_request(self, model_id: str, question: str, max_tokens: int) -> Tuple[str, bool, bool]:
        """Make request to Groq API"""
        api_key = self.api_keys.get('groq')
        if not api_key:
            return "", False, False
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 429:
            return "", False, True  # Rate limited
        
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip(), True, False
    
    def _openrouter_request(self, model_id: str, question: str, max_tokens: int) -> Tuple[str, bool, bool]:
        """Make request to OpenRouter API"""
        api_key = self.api_keys.get('openrouter')
        if not api_key:
            return "", False, False
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 429:
            return "", False, True  # Rate limited
        
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip(), True, False
    
    def _calculate_delay(self, attempt: int, provider_name: str = "", model_name: str = "") -> float:
        """Calculate delay for exponential backoff with detailed logging"""
        import random
        
        # Calculate base delay before exponential increase
        base_delay = self.retry_config.base_delay
        exponential_multiplier = self.retry_config.exponential_base ** attempt
        raw_delay = base_delay * exponential_multiplier
        
        # Apply max delay cap
        capped_delay = min(raw_delay, self.retry_config.max_delay)
        
        # Apply jitter if enabled (80% to 100% of calculated delay)
        final_delay = capped_delay
        if self.retry_config.jitter:
            jitter_factor = (0.8 + random.random() * 0.2)
            final_delay = capped_delay * jitter_factor
        
        return final_delay
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset usage statistics"""
        for key in self.stats:
            self.stats[key] = 0

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Fallback LLM System - Multi-provider LLM client with intelligent fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fallback_llm.py "What is machine learning?"
  python fallback_llm.py "Explain quantum computing" --max-tokens 500
  python fallback_llm.py "Write a Python function" --provider groq
  python fallback_llm.py "Summarize this text" --max-attempts 3 --base-delay 1.0
        """
    )
    
    parser.add_argument(
        'question',
        help='Question to ask the LLM'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum tokens in response (default: 500)'
    )
    
    parser.add_argument(
        '--provider',
        choices=['cerebras', 'groq', 'openrouter'],
        help='Specific provider to use (default: use fallback order)'
    )
    
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=5,
        help='Maximum retry attempts per model (default: 5)'
    )
    
    parser.add_argument(
        '--base-delay',
        type=float,
        default=2.0,
        help='Base delay for exponential backoff in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--max-delay',
        type=float,
        default=60.0,
        help='Maximum delay for exponential backoff in seconds (default: 60.0)'
    )
    
    parser.add_argument(
        '--disable-jitter',
        action='store_true',
        help='Disable jitter in exponential backoff'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show usage statistics after response'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create retry configuration
    retry_config = RetryConfig(
        max_attempts=args.max_attempts,
        base_delay=args.base_delay,
        max_delay=args.max_delay,
        exponential_base=2.0,
        jitter=not args.disable_jitter
    )
    
    # Initialize LLM
    if args.verbose:
        print("🚀 Initializing Fallback LLM System...")
    
    llm = FallbackLLM(retry_config=retry_config)
    
    # Check API keys
    available_providers = [p for p in llm.fallback_order if llm.api_keys.get(p)]
    if not available_providers:
        print("❌ No API keys found! Please set environment variables:")
        print("   CEREBRAS_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY")
        print("   Or create a .env file with these keys")
        sys.exit(1)
    
    if args.verbose:
        print(f"✅ Available providers: {', '.join(available_providers)}")
    
    # Ask question
    print(f"❓ Question: {args.question}")
    print("🤔 Thinking...")
    
    start_time = time.time()
    response = llm.ask(args.question, args.max_tokens, args.provider)
    end_time = time.time()
    
    # Display response
    if response:
        print(f"\n💡 Response:")
        print(f"{response}")
        print(f"\n⏱️  Response time: {end_time - start_time:.2f}s")
    else:
        print("\n❌ No response received from any provider")
    
    # Show statistics if requested
    if args.stats:
        stats = llm.get_stats()
        print(f"\n📊 Usage Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    main()
