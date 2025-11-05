# 🃏 Vegas Solitaire Streamlit App - Deployment Guide

This guide explains how to run and deploy the Vegas Solitaire Streamlit web application.

## 📋 Table of Contents

1. [Local Development](#local-development)
2. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Features](#features)
5. [Troubleshooting](#troubleshooting)

---

## 🏠 Local Development

### Prerequisites

- Python 3.11+
- pip

### Installation

1. **Clone the repository** (if not already cloned):
   ```bash
   git clone <your-repo-url>
   cd Vegas-Solitaire-Optimization
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Quick Start Commands

```bash
# Run with specific port
streamlit run app.py --server.port 8502

# Run with auto-reload on file changes
streamlit run app.py --server.runOnSave true

# Run with different theme
streamlit run app.py --theme.primaryColor "#ff4b4b"
```

---

## ☁️ Streamlit Cloud Deployment

The easiest way to deploy and share your app publicly!

### Step 1: Prepare Your Repository

1. Ensure your repo is pushed to GitHub
2. Make sure these files are present:
   - `app.py` (main entry point)
   - `requirements.txt` (dependencies)
   - `.streamlit/config.toml` (configuration)

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set main file path: `app.py`
6. Click "Deploy"!

Your app will be live at: `https://<username>-vegas-solitaire-optimization-app-<hash>.streamlit.app`

### Step 3: Configure Settings (Optional)

In Streamlit Cloud dashboard:
- **Secrets**: Add any API keys or secrets
- **Python version**: Set to 3.11+
- **Advanced settings**: Adjust memory/CPU if needed

### Updating the App

Streamlit Cloud automatically redeploys when you push to your GitHub branch!

```bash
git add .
git commit -m "Update app"
git push
```

---

## 🐳 Docker Deployment

For self-hosted deployments using Docker.

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t vegas-solitaire-app .

# Run container
docker run -p 8501:8501 vegas-solitaire-app
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./benchmark_results:/app/benchmark_results
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## ✨ Features

### 🏠 Home Page
- Overview of the project
- Solver descriptions
- Quick stats

### 🎮 Play Mode
- **Interactive gameplay**: Click to make moves
- **Undo/Reset**: Experiment freely
- **Move suggestions**: See all valid moves grouped by type
- **Real-time scoring**: Track your progress

### 👀 Watch Mode
- **Watch AI play**: See solvers in action
- **Speed controls**: Slow/Normal/Fast/Very Fast
- **Multiple solvers**: Random, Heuristic, MCTS
- **Step through**: Move-by-move analysis
- **Auto-play**: Watch full games automatically

### 📊 Benchmark Mode
- **Compare solvers**: Run head-to-head comparisons
- **Configurable**: Adjust number of games, seeds, max moves
- **Detailed statistics**: Win rates, scores, times
- **Visualizations**: Interactive charts and graphs
- **Progress tracking**: Real-time updates

### 📈 Analysis Mode
- **Game analysis**: Analyze specific seeds
- **Solver comparison**: See how solvers handle same game
- **Strategy insights**: Understand solver behavior
- **Statistics**: Deep dive into game theory

---

## 🎨 Customization

### Themes

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"  # Blue accent color
backgroundColor = "#ffffff"  # White background
secondaryBackgroundColor = "#f0f2f6"  # Light gray
textColor = "#262730"  # Dark text
font = "sans serif"
```

### Configuration

Adjust server settings in `.streamlit/config.toml`:

```toml
[server]
port = 8501
maxUploadSize = 200
enableCORS = false
```

---

## 🐛 Troubleshooting

### Issue: "Module not found" error

**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

### Issue: Port already in use

**Solution**: Use a different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: App is slow/laggy

**Solutions**:
1. Reduce number of benchmark games
2. Lower MCTS simulations (use 100 instead of 1000)
3. Use "Run to End" instead of auto-play for full games

### Issue: Cards not displaying properly

**Solution**: Check browser compatibility (use Chrome/Firefox/Safari)

### Issue: Streamlit Cloud deployment fails

**Solutions**:
1. Check `requirements.txt` is complete
2. Verify Python version compatibility
3. Check Streamlit Cloud logs for specific errors
4. Ensure `app.py` is at repository root

---

## 📊 Performance Tips

### Local Development

1. **Cache expensive operations**: Streamlit caching is already implemented
2. **Limit benchmark games**: Start with 10 games, scale up as needed
3. **Use faster solvers for testing**: Random and Heuristic are much faster than MCTS

### Production Deployment

1. **Resource allocation**:
   - Minimum: 1 CPU, 1GB RAM
   - Recommended: 2 CPU, 2GB RAM
   - For heavy benchmarks: 4 CPU, 4GB RAM

2. **Optimize MCTS**:
   - Default: 100 simulations (good balance)
   - High quality: 1000 simulations (slower)
   - Quick demos: 50 simulations (faster)

---

## 🚀 Quick Demo

Want to show off the app quickly?

1. **Start the app**: `streamlit run app.py`
2. **Go to Watch Mode**: Click "👀 Watch Mode" in sidebar
3. **Select MCTS(100)**: Choose from dropdown
4. **Click "Run to End"**: Watch a complete game in ~10 seconds
5. **Go to Benchmark**: Run comparison with 10 games

---

## 📝 Environment Variables

Optional environment variables for configuration:

```bash
# Set default port
export STREAMLIT_SERVER_PORT=8501

# Set theme
export STREAMLIT_THEME_PRIMARY_COLOR="#1f77b4"

# Disable telemetry
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud](https://share.streamlit.io)
- [Streamlit Forum](https://discuss.streamlit.io)
- [Project Repository](https://github.com/your-username/Vegas-Solitaire-Optimization)

---

## 📞 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review Streamlit logs: `~/.streamlit/logs/`
3. Open an issue on GitHub
4. Check Streamlit Community Forum

---

## 🎉 Success!

Once deployed, your app will be accessible to anyone with the URL. Share it and let others play with your AI solvers!

Example URL: `https://your-app.streamlit.app`

Happy deploying! 🚀
