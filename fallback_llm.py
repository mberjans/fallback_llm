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
from datetime import datetime

try:
    from cloudflare import Cloudflare
    CLOUDFLARE_AVAILABLE = True
except ImportError:
    CLOUDFLARE_AVAILABLE = False

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

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    provider: str
    model_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    last_tested: Optional[str] = None

    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

class FallbackLLM:
    """Multi-provider LLM client with intelligent fallback"""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """Initialize the fallback LLM system"""
        self.retry_config = retry_config or RetryConfig()
        self.fallback_order = ['cerebras', 'groq', 'cloudflare', 'openrouter']
        self.provider_health = {
            provider: ProviderHealth() for provider in self.fallback_order
        }
        
        # Load API keys from environment
        self.api_keys = self._load_api_keys()

        # Store Cloudflare account ID
        self.cloudflare_account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'provider_switches': 0
        }

        # Model performance tracking
        self.model_performance = {}
        
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
            'cloudflare': [
                '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b',
                '@cf/mistral/mistral-small-3.1-24b-instruct',
                '@cf/meta/llama-4-scout-17b-16e-instruct',
                '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
                '@cf/meta/llama-3.1-70b-instruct',
                '@cf/qwen/qwq-32b'
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
        api_keys['cloudflare'] = os.getenv('CLOUDFLARE_API_TOKEN')
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
        # Initialize model performance tracking if not exists
        model_key = f"{provider}:{model_id}"
        if model_key not in self.model_performance:
            self.model_performance[model_key] = ModelPerformance(provider, model_id)

        perf = self.model_performance[model_key]
        perf.total_requests += 1

        start_time = time.time()
        try:
            if provider == "cerebras":
                response, success, is_rate_limit = self._cerebras_request(model_id, question, max_tokens)
            elif provider == "groq":
                response, success, is_rate_limit = self._groq_request(model_id, question, max_tokens)
            elif provider == "cloudflare":
                response, success, is_rate_limit = self._cloudflare_request(model_id, question, max_tokens)
            elif provider == "openrouter":
                response, success, is_rate_limit = self._openrouter_request(model_id, question, max_tokens)
            else:
                response, success, is_rate_limit = "", False, False

            end_time = time.time()
            response_time = end_time - start_time

            # Update performance metrics
            if success:
                perf.successful_requests += 1
                perf.total_response_time += response_time
                perf.min_response_time = min(perf.min_response_time, response_time)
                perf.max_response_time = max(perf.max_response_time, response_time)
            else:
                perf.failed_requests += 1

            perf.last_tested = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return response, success, is_rate_limit

        except Exception as e:
            end_time = time.time()
            perf.failed_requests += 1
            perf.last_tested = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

    def _cloudflare_request(self, model_id: str, question: str, max_tokens: int) -> Tuple[str, bool, bool]:
        """Make request to Cloudflare Workers AI"""
        if not CLOUDFLARE_AVAILABLE:
            return "", False, False

        api_token = self.api_keys.get('cloudflare')
        if not api_token or not self.cloudflare_account_id:
            return "", False, False

        try:
            # Initialize Cloudflare client
            client = Cloudflare(api_token=api_token)

            # Prepare messages in the format expected by Cloudflare
            messages = [{"role": "user", "content": question}]

            # Make the API call
            response = client.ai.run(
                account_id=self.cloudflare_account_id,
                model_name=model_id,
                messages=messages
            )

            # Extract the response content
            # Cloudflare returns a dict directly
            if isinstance(response, dict):
                if 'response' in response:
                    content = response['response']
                    return content.strip(), True, False
                elif 'content' in response:
                    content = response['content']
                    return content.strip(), True, False
                elif 'text' in response:
                    content = response['text']
                    return content.strip(), True, False
                else:
                    # Fallback: convert entire response to string
                    content = str(response)
                    return content.strip(), True, False
            elif hasattr(response, 'result'):
                # Handle object-style response
                if hasattr(response.result, 'response'):
                    content = response.result.response
                    return content.strip(), True, False
                elif isinstance(response.result, dict):
                    if 'response' in response.result:
                        content = response.result['response']
                        return content.strip(), True, False

            return "", False, False

        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or 'too many requests' in error_msg:
                return "", False, True  # Rate limited
            else:
                print(f"❌ Cloudflare API error: {e}")
                return "", False, False

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

    def test_all_models_and_providers(self, test_question: str = "What is 2+2?", max_tokens: int = 50) -> Dict:
        """Test all models across all providers and return performance data"""
        print("🧪 Testing all models and providers...")
        print("=" * 60)

        results = {}
        total_models = sum(len(models) for models in self.models.values())
        current_model = 0

        for provider_name in self.fallback_order:
            if not self.api_keys.get(provider_name):
                print(f"⚠️  Skipping {provider_name} - no API key available")
                continue

            print(f"\n🔍 Testing {provider_name} provider...")
            provider_models = self.models.get(provider_name, [])

            for model_id in provider_models:
                current_model += 1
                print(f"   🎯 Testing model {current_model}/{total_models}: {model_id}")

                # Test this specific model
                start_time = time.time()
                response, success, is_rate_limit = self._make_api_request(
                    provider_name, model_id, test_question, max_tokens
                )
                end_time = time.time()

                status = "✅ Success" if success else ("⏳ Rate Limited" if is_rate_limit else "❌ Failed")
                response_time = end_time - start_time
                print(f"      {status} - {response_time:.2f}s")

                # Small delay between requests to be respectful
                time.sleep(0.5)

        return self.model_performance

    def generate_markdown_report(self, output_file: str = "model_performance_report.md") -> str:
        """Generate a comprehensive markdown report of all model and provider performance"""

        # Group performance data by provider
        provider_data = {}
        for model_key, perf in self.model_performance.items():
            provider = perf.provider
            if provider not in provider_data:
                provider_data[provider] = []
            provider_data[provider].append(perf)

        # Generate markdown content
        report_lines = []
        report_lines.append("# Fallback LLM System - Model & Provider Performance Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("## Executive Summary")
        report_lines.append("")

        # Calculate overall statistics
        total_models_tested = len(self.model_performance)
        total_requests = sum(perf.total_requests for perf in self.model_performance.values())
        total_successful = sum(perf.successful_requests for perf in self.model_performance.values())
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0

        report_lines.append(f"- **Total Models Tested:** {total_models_tested}")
        report_lines.append(f"- **Total Requests:** {total_requests}")
        report_lines.append(f"- **Overall Success Rate:** {overall_success_rate:.1f}%")
        report_lines.append(f"- **Providers Available:** {len(provider_data)}")
        report_lines.append("")

        # Provider-by-provider analysis
        report_lines.append("## Provider Performance Analysis")
        report_lines.append("")

        for provider_name in self.fallback_order:
            if provider_name not in provider_data:
                continue

            models = provider_data[provider_name]
            report_lines.append(f"### {provider_name.title()} Provider")
            report_lines.append("")

            # Provider summary
            provider_requests = sum(m.total_requests for m in models)
            provider_successful = sum(m.successful_requests for m in models)
            provider_success_rate = (provider_successful / provider_requests * 100) if provider_requests > 0 else 0
            avg_response_time = sum(m.average_response_time for m in models if m.successful_requests > 0) / len([m for m in models if m.successful_requests > 0]) if any(m.successful_requests > 0 for m in models) else 0

            report_lines.append(f"**Provider Summary:**")
            report_lines.append(f"- Models Available: {len(models)}")
            report_lines.append(f"- Total Requests: {provider_requests}")
            report_lines.append(f"- Success Rate: {provider_success_rate:.1f}%")
            report_lines.append(f"- Average Response Time: {avg_response_time:.2f}s")
            report_lines.append("")

            # Model details table
            report_lines.append("| Model | Requests | Success Rate | Avg Response Time | Min Time | Max Time | Last Tested |")
            report_lines.append("|-------|----------|--------------|-------------------|----------|----------|-------------|")

            # Sort models by success rate and response time
            sorted_models = sorted(models, key=lambda m: (m.success_rate, -m.average_response_time), reverse=True)

            for model in sorted_models:
                min_time = f"{model.min_response_time:.2f}s" if model.min_response_time != float('inf') else "N/A"
                max_time = f"{model.max_response_time:.2f}s" if model.max_response_time > 0 else "N/A"
                avg_time = f"{model.average_response_time:.2f}s" if model.average_response_time > 0 else "N/A"
                last_tested = model.last_tested or "Never"

                report_lines.append(f"| `{model.model_id}` | {model.total_requests} | {model.success_rate:.1f}% | {avg_time} | {min_time} | {max_time} | {last_tested} |")

            report_lines.append("")

        # Performance rankings
        report_lines.append("## Performance Rankings")
        report_lines.append("")

        # Fastest models
        successful_models = [perf for perf in self.model_performance.values() if perf.successful_requests > 0]
        if successful_models:
            report_lines.append("### 🚀 Fastest Models (by average response time)")
            report_lines.append("")
            fastest_models = sorted(successful_models, key=lambda m: m.average_response_time)[:10]

            report_lines.append("| Rank | Provider | Model | Avg Response Time | Success Rate |")
            report_lines.append("|------|----------|-------|-------------------|--------------|")

            for i, model in enumerate(fastest_models, 1):
                report_lines.append(f"| {i} | {model.provider.title()} | `{model.model_id}` | {model.average_response_time:.2f}s | {model.success_rate:.1f}% |")

            report_lines.append("")

        # Most reliable models
        if successful_models:
            report_lines.append("### 🎯 Most Reliable Models (by success rate)")
            report_lines.append("")
            most_reliable = sorted(successful_models, key=lambda m: (m.success_rate, -m.average_response_time), reverse=True)[:10]

            report_lines.append("| Rank | Provider | Model | Success Rate | Avg Response Time |")
            report_lines.append("|------|----------|-------|--------------|-------------------|")

            for i, model in enumerate(most_reliable, 1):
                report_lines.append(f"| {i} | {model.provider.title()} | `{model.model_id}` | {model.success_rate:.1f}% | {model.average_response_time:.2f}s |")

            report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")

        if successful_models:
            fastest = min(successful_models, key=lambda m: m.average_response_time)
            most_reliable = max(successful_models, key=lambda m: m.success_rate)

            report_lines.append(f"- **For Speed:** Use `{fastest.provider}` with model `{fastest.model_id}` (avg: {fastest.average_response_time:.2f}s)")
            report_lines.append(f"- **For Reliability:** Use `{most_reliable.provider}` with model `{most_reliable.model_id}` (success rate: {most_reliable.success_rate:.1f}%)")

            # Best overall (balance of speed and reliability)
            best_overall = max(successful_models, key=lambda m: m.success_rate / (m.average_response_time + 0.1))
            report_lines.append(f"- **Best Overall:** Use `{best_overall.provider}` with model `{best_overall.model_id}` (balanced performance)")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*Report generated by Fallback LLM System*")

        # Write to file
        report_content = "\n".join(report_lines)
        with open(output_file, 'w') as f:
            f.write(report_content)

        return report_content

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
  python fallback_llm.py "Explain AI concepts" --provider cloudflare
  python fallback_llm.py "Summarize this text" --max-attempts 3 --base-delay 1.0
  python fallback_llm.py "Test question" --all-model-provider-report --verbose
  python fallback_llm.py "Custom test" --all-model-provider-report --report-output my_report.md
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
        choices=['cerebras', 'groq', 'cloudflare', 'openrouter'],
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

    parser.add_argument(
        '--all-model-provider-report',
        action='store_true',
        help='Test all models and providers, generate comprehensive markdown performance report'
    )

    parser.add_argument(
        '--report-output',
        type=str,
        default='model_performance_report.md',
        help='Output file for the performance report (default: model_performance_report.md)'
    )

    parser.add_argument(
        '--test-question',
        type=str,
        default='What is 2+2?',
        help='Question to use for testing all models (default: "What is 2+2?")'
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
    available_providers = []
    for p in llm.fallback_order:
        if p == 'cloudflare':
            # Cloudflare needs both API token and account ID
            if llm.api_keys.get(p) and llm.cloudflare_account_id and CLOUDFLARE_AVAILABLE:
                available_providers.append(p)
        else:
            if llm.api_keys.get(p):
                available_providers.append(p)

    if not available_providers:
        print("❌ No API keys found! Please set environment variables:")
        print("   CEREBRAS_API_KEY, GROQ_API_KEY, CLOUDFLARE_API_TOKEN+CLOUDFLARE_ACCOUNT_ID, OPENROUTER_API_KEY")
        print("   Or create a .env file with these keys")
        if not CLOUDFLARE_AVAILABLE:
            print("   Note: Install 'cloudflare' package for Cloudflare Workers AI support")
        sys.exit(1)
    
    if args.verbose:
        print(f"✅ Available providers: {', '.join(available_providers)}")

    # Handle comprehensive model and provider report
    if args.all_model_provider_report:
        print("🚀 Starting comprehensive model and provider performance testing...")
        print("This may take several minutes to complete.\n")

        # Test all models and providers
        llm.test_all_models_and_providers(
            test_question=args.test_question,
            max_tokens=args.max_tokens
        )

        # Generate markdown report
        print(f"\n📝 Generating performance report...")
        report_content = llm.generate_markdown_report(args.report_output)

        print(f"✅ Performance report saved to: {args.report_output}")
        print(f"📊 Report contains data for {len(llm.model_performance)} models")

        # Show brief summary
        if args.verbose:
            total_requests = sum(perf.total_requests for perf in llm.model_performance.values())
            total_successful = sum(perf.successful_requests for perf in llm.model_performance.values())
            overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0

            print(f"\n📈 Quick Summary:")
            print(f"   Total models tested: {len(llm.model_performance)}")
            print(f"   Total requests made: {total_requests}")
            print(f"   Overall success rate: {overall_success_rate:.1f}%")

        return

    # Regular question asking mode
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
