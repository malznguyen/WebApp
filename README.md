# Enhanced Toolkit v2.0

Enhanced Toolkit is a Python and JavaScript application for reverse image search, AI-driven image description, document summarization, and metadata analysis. It provides a browser-based interface through the Eel framework.

## Features Overview
- Reverse image search via SERP API with optional social media filtering
- AI-powered image description using the OpenAI Vision API
- Document summarization across multiple models (DeepSeek, Grok, ChatGPT)
- Image metadata extraction and analysis
- Web interface built with Tailwind CSS, Alpine.js, and Chart.js

## Technical Requirements
- Python 3.8 or higher
- pip for dependency management
- SERP API key and Imgur client ID for image search
- Optional API keys for OpenAI, DeepSeek, and Grok

## Installation & Setup
```bash
# Clone and install dependencies
git clone <repo-url>
cd WebApp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in the project root:
```ini
SERP_API_KEY=your-serp-api-key
IMGUR_CLIENT_ID=your-imgur-client-id
DEEPSEEK_API_KEY=optional-deepseek-key
GROK_API_KEY=optional-grok-key
CHATGPT_API_KEY=optional-openai-key
```

## Usage Examples
Run the application locally:
```bash
python main.py
```

Example API call using Eel exposed functions:
```python
import eel

# Retrieve configuration
config = eel.get_app_config()()
print(config)
```

## API Documentation
Key Eel endpoints:
- `get_app_config()` – Returns configured API status
- `search_image_web(base64_image, filename, social_only)` – Reverse image search
- `describe_image_web(base64_image, filename, language, detail_level, prompt)` – Image description
- `process_document_web(path, settings)` – Summarize document
- `process_document_async_web(file_or_list, settings)` – Async document summarization

## Architecture Overview
```
+------------+       +--------------+       +--------------+
| Frontend   | <---> |  Eel Bridge  | <---> |  Python Core |
+------------+       +--------------+       +--------------+
       |                                     |
   Tailwind/                         SERP API, Imgur,
   Alpine.js                         OpenAI, etc.
```
The frontend communicates with Python modules through Eel. Background operations use threading for non-blocking searches and document processing.

## Security & Privacy
- Store all API keys in environment variables or a `.env` file and never commit them to source control.
- Uploaded images and documents are processed in temporary directories and removed after completion.
- Network requests respect configured timeouts to avoid hanging connections.

## Deployment
Use a production-ready web server or create a desktop build with PyInstaller. When deploying, ensure that API keys are provided via environment variables and that file permissions restrict access to uploaded data.

## Contributing
1. Fork the repository and create a feature branch.
2. Follow PEP 8 for Python code and standard ESLint rules for JavaScript.
3. Run unit tests before submitting a pull request.

## License & Legal
This project is provided under the MIT License. See `LICENSE` if included in the repository.

## Troubleshooting
- **Invalid API keys** – Ensure keys in `.env` are correct. The application logs errors when API validation fails.
- **Module import errors** – Verify all dependencies are installed and that your virtual environment is active.

## Performance Considerations
- Large images (>5MB) are resized before uploading to reduce latency.
- Document processing is parallelized with a limited worker count to balance CPU usage.

## Known Limitations
- OpenAI Vision requires valid credentials and may incur usage costs.
- Extremely large documents may be truncated to stay within API limits.
