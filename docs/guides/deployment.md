# 🚀 Deployment Guide - NBA Predictor Analytics

## Overview

This guide covers deployment options for the NBA Predictor Analytics Streamlit dashboard, including local development, cloud deployment, and production best practices.

## 🏗️ System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **Memory**: 2GB RAM minimum
- **Storage**: 1GB free space
- **Network**: Internet connection for API access

### Recommended Production
- **Python**: 3.11+ with virtual environment
- **Memory**: 4GB+ RAM
- **Storage**: 5GB+ free space
- **Network**: Stable connection with API rate limit consideration

## 🛠️ Local Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/nba-predictor-streamlit.git
cd nba-predictor-streamlit

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file for API keys:
```bash
# The Odds API (optional - system works without it)
THE_ODDS_API_KEY=your_api_key_here

# NBA Season configuration
NBA_SEASON=2025-26

# Debug mode (optional)
DEBUG_MODE=false
```

### 3. Local Development Server

```bash
# Basic development server
streamlit run main_app.py

# Development with hot reload
streamlit run main_app.py --server.runOnSave true

# Custom port and host
streamlit run main_app.py --server.port 8501 --server.address 0.0.0.0

# Development with debugging
streamlit run main_app.py --server.runOnSave true --server.headless false
```

**Access**: http://localhost:8501

## 🌐 Cloud Deployment Options

### Option 1: Streamlit Community Cloud

**Recommended for**: Easy deployment, public projects, small teams

#### Step-by-Step Deployment

1. **Prepare Repository**
```bash
# Ensure all dependencies are in requirements.txt
pip freeze > requirements.txt

# Commit changes
git add .
git commit -m "Add deployment configuration"
git push origin main
```

2. **Streamlit Cloud Setup**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub account
- Select repository: `yourusername/nba-predictor-streamlit`
- Main file path: `main_app.py`
- Python version: 3.11+

3. **Environment Variables**
In Streamlit Cloud dashboard:
- `THE_ODDS_API_KEY`: Your The Odds API key (optional)
- `NBA_SEASON`: `2025-26`
- `PYTHONPATH`: `/mount/src`

4. **Advanced Settings**
- **Memory**: 2GB minimum (recommended 4GB)
- **CPU**: 2 cores minimum
- **Timeout**: 30 minutes (for API calls)

#### Pros & Cons

**Pros:**
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Easy GitHub integration
- ✅ Managed environment
- ✅ No server maintenance

**Cons:**
- ❌ Limited customization
- ❌ Resource constraints
- ❌ No custom domains on free tier
- ❌ Limited background processes

### Option 2: Railway

**Recommended for**: Production applications, custom domains

#### Deployment Steps

1. **Install Railway CLI**
```bash
npm install -g @railway/cli
railway login
```

2. **Initialize Railway Project**
```bash
railway init
railway up
```

3. **Configure Service**
Create `railway.toml`:
```toml
[build]
builder = "nixpacks"

[deploy]
healthcheckPath = "/"
healthcheckTimeout = 100
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 10

[[services]]
name = "nba-predictor"

[services.variables]
PORT = "8501"
THE_ODDS_API_KEY = "${THE_ODDS_API_KEY}"
NBA_SEASON = "2025-26"
```

4. **Deploy**
```bash
railway deploy
```

### Option 3: Docker Deployment

**Recommended for**: Full control, production environments

#### 1. Create Dockerfile

```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start the application
CMD ["streamlit", "run", "main_app.py"]
```

#### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  nba-predictor:
    build: .
    ports:
      - "8501:8501"
    environment:
      - THE_ODDS_API_KEY=${THE_ODDS_API_KEY}
      - NBA_SEASON=2025-26
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_PORT=8501
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    volumes:
      - ./data:/app/data
    networks:
      - nba-network

networks:
  nba-network:
    driver: bridge
```

#### 3. Deploy with Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 4: VPS/Cloud Server

**Recommended for**: Full control, high traffic, custom requirements

#### Server Setup (Ubuntu 22.04)

1. **System Preparation**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install other dependencies
sudo apt install git nginx certbot python3-certbot-nginx -y
```

2. **Application Setup**
```bash
# Create application user
sudo useradd -m -s /bin/bash nbaapp
sudo su - nbaapp

# Clone and setup application
git clone https://github.com/yourusername/nba-predictor-streamlit.git
cd nba-predictor-streamlit

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Systemd Service**
Create `/etc/systemd/system/nba-predictor.service`:
```ini
[Unit]
Description=NBA Predictor Streamlit App
After=network.target

[Service]
Type=simple
User=nbaapp
WorkingDirectory=/home/nbaapp/nba-predictor-streamlit
Environment=PATH=/home/nbaapp/nba-predictor-streamlit/.venv/bin
ExecStart=/home/nbaapp/nba-predictor-streamlit/.venv/bin/streamlit run main_app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

4. **Nginx Reverse Proxy**
Create `/etc/nginx/sites-available/nba-predictor`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # WebSocket support
    location /_stcore/stream {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

5. **SSL Certificate**
```bash
sudo certbot --nginx -d your-domain.com
```

6. **Start Services**
```bash
sudo systemctl enable nba-predictor
sudo systemctl start nba-predictor
sudo systemctl enable nginx
sudo systemctl start nginx
```

## 🔧 Production Configuration

### Environment Variables

Create production `.env` file:
```bash
# API Configuration
THE_ODDS_API_KEY=your_production_api_key
NBA_SEASON=2025-26

# Security
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Performance
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_SERVER_ENABLE_CACHING=true
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Logging
STREAMLIT_LOGGER_LEVEL=info
```

### Streamlit Configuration

Create `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false

[logger]
level = "info"

[theme]
base = "dark"
primaryColor = "#FF6B35"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

## 📊 Monitoring & Maintenance

### Health Checks

```bash
# Application health check
curl -f http://localhost:8501/_stcore/health

# Service status
sudo systemctl status nba-predictor

# Resource usage
htop
```

### Log Management

```bash
# View application logs
sudo journalctl -u nba-predictor -f

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Performance Monitoring

```bash
# Resource monitoring
docker stats  # For Docker deployments
free -h       # Memory usage
df -h         # Disk usage
```

### Backup Strategy

```bash
# Backup application and data
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf backup_$DATE.tar.gz \
    /home/nbaapp/nba-predictor-streamlit \
    /etc/nginx/sites-available/nba-predictor \
    /etc/systemd/system/nba-predictor.service
```

## 🚨 Troubleshooting Production Issues

### Common Problems

#### API Rate Limiting
**Symptoms**: Errors about quota exceeded
**Solutions**:
- Monitor API usage in System Status tab
- Implement request caching
- Consider upgrading API plan
- Use fallback data sources

#### Memory Issues
**Symptoms**: Out of memory errors, slow performance
**Solutions**:
- Increase server RAM
- Implement memory caching limits
- Monitor with `htop` or `docker stats`
- Restart services periodically

#### Database Connection Issues
**Symptoms**: Failed to connect to APIs
**Solutions**:
- Check network connectivity
- Verify API keys are valid
- Monitor API service status
- Check firewall settings

#### SSL Certificate Issues
**Symptoms**: HTTPS errors, certificate warnings
**Solutions**:
```bash
# Renew certificates
sudo certbot renew

# Test certificate
sudo certbot certificates

# Force renewal
sudo certbot renew --force-renewal
```

### Performance Optimization

#### Caching Strategy
- Enable Streamlit caching
- Cache API responses
- Use CDN for static assets
- Implement session caching

#### Resource Optimization
- Monitor memory usage
- Optimize API call frequency
- Use connection pooling
- Implement request batching

## 📈 Scaling Considerations

### Horizontal Scaling
- Load balancer configuration
- Multiple application instances
- Database connection pooling
- Session management

### Vertical Scaling
- Increase server resources
- Optimize database queries
- Implement caching layers
- Use faster storage

### Geographic Distribution
- CDN implementation
- Multi-region deployment
- Database replication
- API endpoint optimization

---

## 🔗 Additional Resources

- [Streamlit Deployment Documentation](https://docs.streamlit.io/knowledge-base/tutorials/deploy)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Nginx Configuration Guide](https://nginx.org/en/docs/)
- [Systemd Service Management](https://www.freedesktop.org/software/systemd/man/systemd.service.html)

**🎯 For production deployments, always test thoroughly in a staging environment first!**