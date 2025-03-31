# LLM Inference Framework

A flexible and extensible framework for running inference with various Large Language Models (LLMs). This framework provides a unified interface for interacting with different LLM providers and model types, including OpenAI's GPT models, Hugging Face models, and VLLM-based models.

## Features

- **Unified Interface**: Consistent API for interacting with different LLM providers
- **Multiple Model Support**:
  - OpenAI GPT models (GPT-3.5, GPT-4)
  - Hugging Face models (CodeLlama, Mistral, Vicuna, etc.)
  - VLLM-based models
  - WizardLM models
  - Salesforce models
  - Incoder models
- **Flexible Configuration**: YAML-based configuration for model parameters
- **Batch Processing**: Support for processing multiple prompts efficiently
- **Memory Management**: Efficient handling of model loading and inference
- **Token Usage Tracking**: Monitor token consumption across different models

## Installation

```bash
git clone https://github.com/yourusername/llm_inference.git
cd llm_inference
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from chat_session.selector import select_chat_model

# Load configuration
config = {
    'model_cache': '/path/to/model/cache',
    'max_length': 4096,
    'num_output_tokens': 1024,
    'batch_size': {
        'default': 1,
        'model_name': 4  # Specific batch size for a model
    }
}

# Initialize model
model = select_chat_model(config, 'gpt-3.5-turbo')

# Get response
response = model.get_response("Your prompt here")
```

### Configuration

The framework uses a YAML configuration file to manage model parameters. Example configuration:

```yaml
model_cache: /path/to/model/cache
max_length: 4096
num_output_tokens: 1024
batch_size:
  default: 1
  gpt-3.5-turbo: 4
  codellama/CodeLlama-7b-hf: 2
temperature: 0.1
top_p: 0.95
top_k: 1
num_beams: 1
```

## Supported Models

### OpenAI Models
- GPT-3.5 Turbo
- GPT-4

### Hugging Face Models
- CodeLlama (7B, 13B, 34B)
- Mistral (7B)
- Vicuna (7B, 13B)
- Phi models
- CodeGemma
- And many more...

### VLLM Models
- Mistral
- CodeLlama
- Vicuna
- StarCoder
- And others...

## Project Structure

```
llm_inference/
├── chat_session/
│   ├── __init__.py
│   ├── chat_session.py      # Base chat session class
│   ├── selector.py          # Model selection logic
│   ├── openai_gpt.py        # OpenAI implementation
│   ├── hf_pipeline_models.py # Hugging Face pipeline models
│   ├── hf_generate_models.py # Hugging Face generate models
│   ├── wizardcoder.py       # WizardLM implementation
│   ├── salesforce.py        # Salesforce models
│   ├── vllm_models.py       # VLLM implementation
│   └── incoder.py           # Incoder models
├── prompt_engg.py           # Prompt engineering utilities
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license. See the LICENSE file for details.
