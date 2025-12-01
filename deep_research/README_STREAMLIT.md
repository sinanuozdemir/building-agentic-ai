# 🔍 Deep Research Assistant - Streamlit App

A powerful Streamlit application that provides a user-friendly interface for the Deep Research workflow using LangGraph. This app allows you to perform comprehensive research with real-time streaming events and performance analysis.

## 🚀 Features

- **Interactive Research Interface**: Easy-to-use web interface for defining research objectives
- **Real-time Event Streaming**: Watch the research workflow progress with live updates
- **Performance Analytics**: Visualize workflow performance with interactive charts
- **Configurable Settings**: Adjust maximum research steps and other parameters
- **Download Results**: Export research findings to text files
- **Beautiful UI**: Modern, responsive design with emojis and intuitive navigation

## 📋 Prerequisites

Before running the app, you'll need to obtain API keys for the following services:

1. **OpenRouter API Key**: For accessing OpenAI models
   - Visit: https://openrouter.ai/
   - Sign up and get your API key

2. **Firecrawl API Key**: For web scraping capabilities
   - Visit: https://www.firecrawl.dev/
   - Sign up and get your API key

3. **SERP API Key**: For Google search functionality
   - Visit: https://serpapi.com/
   - Sign up and get your API key

## 🔧 Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd applied-ai-book/deep_research
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r streamlit_requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the `deep_research` directory:
   ```bash
   # OpenRouter API Key (for OpenAI models)
   OPENROUTER_API_KEY=your_openrouter_api_key_here

   # Firecrawl API Key (for web scraping)
   FIRECRAWL_API_KEY=your_firecrawl_api_key_here

   # SERP API Key (for Google search)
   SERP_API_KEY=your_serp_api_key_here

   # Optional: Custom model configurations
   # MODEL_NAME=openai/gpt-4o-mini
   # TEMPERATURE=0.3
   ```

## 🏃 Running the App

1. **Start the Streamlit app**:
   ```bash
   ./run_streamlit_app.sh
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Configure your research settings** in the sidebar

4. **Enter your research objective** and click "🚀 Start Research"

## 📊 App Interface

### Main Components

1. **Research Objective Input**: Large text area for defining your research goal
2. **Live Event Stream**: Real-time display of workflow progress
3. **Final Results**: Comprehensive research summary
4. **Performance Analysis**: Interactive charts and metrics

### Sidebar Features

- **Configuration Status**: Shows if all API keys are properly set
- **Research Settings**: Configure maximum research steps
- **About Section**: Information about the app's capabilities

## 🎯 Example Research Objectives

Try these example objectives to get started:

1. **Market Analysis**:
   ```
   Write a comprehensive analysis of the latest AI developments in 2024, focusing on LLM capabilities and market trends.
   ```

2. **Competitive Research**:
   ```
   Create a detailed comparison between Tesla and traditional automakers in the electric vehicle market, including pricing, technology, and market share.
   ```

3. **Technical Research**:
   ```
   Research the current state of quantum computing, its potential applications, and the major players in the field.
   ```

4. **Industry Report**:
   ```
   Analyze the impact of remote work on commercial real estate markets in major US cities.
   ```

## 📈 Performance Monitoring

The app provides detailed performance analytics including:

- **Event Distribution**: Pie chart showing workflow node activity
- **Timeline Visualization**: Progress over time with event markers
- **Key Metrics**: Total events, execution time, and research steps
- **Real-time Progress**: Live progress bar and status updates

## 🔍 Workflow Overview

The Deep Research Assistant follows this workflow:

1. **Initial Planning**: Generates a research plan based on your objective
2. **Step Execution**: Executes each research step using advanced agents
3. **Replanning**: Adapts the plan based on intermediate findings
4. **Summarization**: Creates a comprehensive final report
5. **Performance Analysis**: Provides insights into the research process

## 🛠️ Troubleshooting

### Common Issues

1. **Missing API Keys**:
   - Check that all required environment variables are set in your `.env` file
   - Verify API keys are valid and have sufficient credits

2. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r streamlit_requirements.txt`
   - Check that you're using the correct Python environment

3. **Slow Performance**:
   - Reduce the maximum number of research steps
   - Check your internet connection
   - Verify API rate limits aren't being exceeded

4. **Streamlit Issues**:
   - Clear browser cache
   - Try refreshing the page
   - Restart the Streamlit server

### Debug Mode

To enable debug mode, add this to your `.env` file:
```bash
DEBUG=True
```

## 🤝 Contributing

To contribute to the Deep Research Assistant:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is part of the Applied AI Book and follows the same license terms.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangGraph](https://python.langchain.com/docs/langgraph)
- Uses [OpenRouter](https://openrouter.ai/) for AI model access
- Web scraping via [Firecrawl](https://www.firecrawl.dev/)
- Search functionality through [SERP API](https://serpapi.com/)

---

**Happy Researching!** 🚀🔍 