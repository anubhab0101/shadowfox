# ShadowFox

A comprehensive Python project featuring web scraping, data processing, and AI integration across different difficulty levels.

## Project Structure

```
shadowfox/
├── ShadowFox/
│   ├── Beginer_level_task/
│   ├── Intermediate_level_task/
│   └── Advance_Level/
│       └── GEMINI_INTERGRATION/
```

## Beginner Level Tasks

Located in `ShadowFox/Beginer_level_task/`:

1. `beginer_level_qn2.1.py` - Basic string manipulation and data processing
2. `beginer_level_qn2.2.py` - File handling and text processing
3. `beginer_level_qn2.3.py` - Simple data structures implementation
4. `beginer_level_qn3.py` - Control flow and conditional statements
5. `beginer_level_qn4.1.py` - Basic function implementation
6. `beginer_level_qn4.2.py` - Error handling and exceptions
7. `beginer_level_qn4.3.py` - Working with lists and dictionaries
8. `beginer_level_qn8.py` - Simple file I/O operations
9. `beginer_level_qn9.py` - Basic algorithm implementation

## Intermediate Level Tasks

Located in `ShadowFox/Intermediate_level_task/`:

1. `task2.1.py` - Web Scraping GUI Application
   - Features:
     - Scrape quotes from websites
     - Extract news headlines
     - Collect metadata
   - Built with:
     - Tkinter for GUI
     - Selenium for web scraping
     - Threading for non-blocking operations
   - Functions:
     - Data extraction
     - Progress tracking
     - Save/export capabilities

2. `task2.2.py` - Advanced data processing and analysis

## Advanced Level - GEMINI Integration

Located in `ShadowFox/Advance_Level/GEMINI_INTERGRATION/`:

### Main Components

1. `app.py` - Streamlit web application
   - Interactive UI for AI-powered analysis
   - Real-time data processing
   - Visualization capabilities

2. Modules:
   - `gemini_client.py` - Google's Generative AI integration
   - `analysis_engine.py` - Data analysis and processing
   - `research_framework.py` - Research methodology implementation
   - `visualization.py` - Data visualization tools

3. Utils:
   - `rate_limiter.py` - API rate limiting implementation
   - `text_processor.py` - Text processing utilities

4. Data:
   - `sample_prompts.py` - Example prompts for AI interaction

## Setup and Installation

1. Install dependencies:
```bash
poetry install
```

2. Configure environment:
```bash
cp .env.example .env
# Add your Google API key to .env
```

3. Run specific components:
- Beginner tasks: Run individual Python files
- Intermediate GUI: `python ShadowFox/Intermediate_level_task/task2.1.py`
- Advanced Streamlit app: `streamlit run ShadowFox/Advance_Level/GEMINI_INTERGRATION/app.py`

## Technologies Used

- Python 3.11+
- Streamlit
- Selenium
- Google Generative AI
- Tkinter
- Plotly
- Pandas

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.