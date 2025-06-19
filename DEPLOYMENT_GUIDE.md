# 🚀 NBA Predictor v2.0.0 - Deployment Guide

## 📋 Release Overview

**Version**: v2.0.0  
**Release Date**: June 19, 2025  
**Repository**: https://github.com/fulvian/nba-predictor-streamlit  

## ✨ Key Features

🏀 **Advanced NBA Game Prediction System**
- Multi-model ML predictions (Regular Season, Playoff, Hybrid)
- Real-time injury impact analysis with dual-source validation
- Monte Carlo simulations (100,000 iterations)
- Professional performance analytics

💰 **Intelligent Betting System**
- Value bet identification with edge calculation
- Kelly criterion stake optimization
- Comprehensive bankroll management
- Risk assessment and performance tracking

📊 **Professional Streamlit Interface**
- Modern, mobile-responsive UI
- Real-time data visualization with Plotly
- Interactive dashboards and analytics
- Data export capabilities

## 🛠 Installation Instructions

### Prerequisites
- Python 3.8+
- Git
- 2GB+ RAM recommended

### Quick Start

```bash
# Clone the repository
git clone https://github.com/fulvian/nba-predictor-streamlit.git
cd nba-predictor-streamlit/autoover5

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Alternative Installation (Local)

```bash
# Download ZIP from GitHub
# Extract to desired directory
# Navigate to autoover5 folder
cd path/to/nba-predictor-streamlit/autoover5

# Install requirements
pip install streamlit pandas plotly numpy scikit-learn joblib requests beautifulsoup4

# Launch app
streamlit run app.py
```

## 🌐 Access Information

- **Local URL**: http://localhost:8501
- **Network URL**: http://[your-ip]:8501

## 📁 Project Structure

```
autoover5/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── data/                          # Data storage
│   ├── models/                    # ML models
│   ├── cache/                     # Cached data
│   └── risultati_bet_completi.csv # Betting results
├── injury_reporter.py             # Injury analysis module
├── momentum_predictor_*.py         # Prediction modules
└── README.md                      # Documentation
```

## ⚙️ Configuration

### Environment Variables (Optional)
```bash
export NBA_API_DELAY=1.0           # API request delay
export STREAMLIT_SERVER_PORT=8501  # Custom port
export STREAMLIT_SERVER_ADDRESS=0.0.0.0  # Network access
```

### Data Sources
- NBA Stats API (official)
- CBS Sports (injury reports)
- ESPN (injury validation)

## 🔧 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
lsof -ti:8501 | xargs kill -9
streamlit run app.py
```

**DateTime Conversion Errors**
- ✅ Fixed in v2.0.0 with robust error handling

**Missing Dependencies**
```bash
pip install --upgrade -r requirements.txt
```

**Model Loading Issues**
- Ensure `models/` directory exists
- Check file permissions
- Verify disk space (500MB+ recommended)

## 📊 System Requirements

### Minimum
- CPU: 2 cores
- RAM: 2GB
- Storage: 1GB
- Network: Stable internet connection

### Recommended
- CPU: 4+ cores
- RAM: 4GB+
- Storage: 2GB+
- Network: High-speed internet

## 🚀 Production Deployment

### Streamlit Cloud
1. Fork repository to your GitHub
2. Connect to Streamlit Cloud
3. Deploy from `autoover5/app.py`

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Heroku
- Add `setup.sh` and `Procfile`
- Deploy via Git or GitHub integration

## 📈 Performance Optimization

- Enable Streamlit caching (already implemented)
- Use data compression for large datasets
- Monitor memory usage during high-load periods
- Consider Redis for production caching

## 🔒 Security Considerations

- API keys should be environment variables
- Enable HTTPS in production
- Implement rate limiting for API calls
- Regular dependency updates

## 📞 Support

- **Repository**: https://github.com/fulvian/nba-predictor-streamlit
- **Issues**: GitHub Issues section
- **Documentation**: README.md files

## 🔄 Version History

- **v2.0.0** (Current): Production-ready release with full features
- **v1.x**: Development versions

## 📝 License

Check repository for license information.

---

**🏀 NBA Predictor v2.0.0 - Ready for Production Deployment! 🚀** 