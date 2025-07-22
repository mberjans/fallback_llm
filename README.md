# Fallback LLM System

A robust, production-ready multi-provider Large Language Model client with intelligent fallback capabilities.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys
Edit the `.env` file with your API keys:
```bash
CEREBRAS_API_KEY=your_cerebras_key_here
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 3. Run as CLI Tool
```bash
python fallback_llm.py "What is machine learning?"
```

### 4. Use as Python Module
```python
from fallback_llm import FallbackLLM

llm = FallbackLLM()
response = llm.ask("What is artificial intelligence?")
print(response)
```

## 📁 Files in this Directory

| File | Description |
|------|-------------|
| `fallback_llm.py` | Main system - CLI tool and Python module |
| `fallback_llm.md` | Comprehensive documentation |
| `example_usage.py` | Usage examples and demonstrations |
| `models_config.json` | LLM models and providers configuration |
| `.env` | API keys configuration (edit with your keys) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file - quick start guide |

## ✨ Key Features

- **🔄 Multi-tier fallback**: Cerebras → Groq → OpenRouter
- **⚡ Model rotation**: Multiple models per provider
- **🔁 Smart retry logic**: Exponential backoff with jitter
- **📊 Monitoring**: Statistics tracking and detailed logging
- **🛠️ Dual interface**: CLI tool and Python module
- **🔧 Configurable**: Custom retry settings and provider selection

## 📖 Documentation

For complete documentation, see [`fallback_llm.md`](fallback_llm.md) which includes:
- Installation and setup
- Command-line flags reference
- Module usage examples
- Configuration options
- Error handling
- Performance characteristics
- Troubleshooting guide

## 🎯 Usage Examples

### Command Line
```bash
# Basic usage
python fallback_llm.py "Explain quantum computing"

# With options
python fallback_llm.py "Write a Python function" --max-tokens 500 --provider groq

# Verbose output with statistics
python fallback_llm.py "Complex question" --verbose --stats
```

### Python Module
```python
from fallback_llm import FallbackLLM, RetryConfig

# Custom configuration
config = RetryConfig(max_attempts=3, base_delay=1.0)
llm = FallbackLLM(retry_config=config)

# Ask questions
response = llm.ask("Your question", max_tokens=300)
```

## 🔧 Configuration

### API Keys
Get your API keys from:
- **Cerebras**: https://inference.cerebras.ai/
- **Groq**: https://console.groq.com/
- **OpenRouter**: https://openrouter.ai/

### Models Configuration
The `models_config.json` file contains detailed information about:
- Available models per provider
- Model capabilities and limits
- Fallback order and priorities
- Default settings

## 📊 Provider Information

| Provider | Speed | Models | Cost |
|----------|-------|--------|------|
| **Cerebras** | Fastest (1-2s) | 4 models | Free tier |
| **Groq** | Very fast (1-3s) | 5 models | Free tier |
| **OpenRouter** | Variable (2-5s) | 4+ models | Free models |

## 🛠️ Development

### Running Examples
```bash
python example_usage.py
```

### Testing Individual Providers
```bash
python fallback_llm.py "Test" --provider cerebras
python fallback_llm.py "Test" --provider groq
python fallback_llm.py "Test" --provider openrouter
```

## 🔍 Troubleshooting

### Common Issues
1. **No API keys**: Set up `.env` file with valid keys
2. **Network errors**: Check internet connection
3. **Rate limiting**: Normal behavior - system handles automatically
4. **Import errors**: Install dependencies with `pip install -r requirements.txt`

### Debug Mode
```bash
python fallback_llm.py "Question" --verbose
```

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements.

---

**For complete documentation and advanced usage, see [`fallback_llm.md`](fallback_llm.md)**
