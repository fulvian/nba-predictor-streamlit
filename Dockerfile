# NBA Predictor Streamlit - Phase 5 Production Dockerfile
# Superpoteri Context7-compliant multi-stage build with PWA features and advanced optimization

# Build stage with Python dependencies
FROM python:3.11-slim as builder

# Set environment variables for Context7 compliance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV CONTEXT7_ENABLED=true
ENV PWA_FEATURES=enabled
ENV SUPERPOTERI_MODE=active

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Context7 compliance labels
LABEL maintainer="NBA Predictor Superpoteri Team" \
      version="5.0.0" \
      context7.compliance="true" \
      context7.pwa.enabled="true" \
      context7.accessibility="WCAG_2_1_AA" \
      context7.responsive="true" \
      context7.realtime="true" \
      context7.ml-operations="true" \
      superpoteri.enabled="true"

# Set environment variables for Context7 production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV ENV=production
ENV CONTEXT7_ENABLED=true
ENV PWA_FEATURES=enabled
ENV SUPERPOTERI_MODE=active
ENV NBA_PREDICTOR_ENV=production
ENV CONTEXT7_RESPONSIVE_DESIGN=true
ENV CONTEXT7_ACCESSIBILITY_FEATURES=true
ENV CONTEXT7_REAL_TIME_UPDATES=true
ENV CONTEXT7_INTELLIGENT_CACHE=true
ENV CONTEXT7_ADVANCED_ML_OPERATIONS=true

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY requirements.txt ./

# Create necessary directories
RUN mkdir -p data logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Create Context7 PWA directories and setup
RUN mkdir -p /app/static/pwa /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user for Context7 security compliance
USER appuser

# Context7 PWA health check with real-time monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Expose port for Context7 PWA features
EXPOSE 8501

# Context7 PWA optimization: Use production command with Superpoteri features
CMD ["streamlit", "run", "src/nba_predictor/streamlit/betting_workflow_dashboard.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.enableCORS=true", \
    "--server.enableXsrfProtection=true", \
    "--browser.gatherUsageStats=false", \
    "--server.enableWebsocketCompression=true", \
    "--server.enableCaching=true", \
    "--server.maxMessageSize=500MB", \
    "--server.enablePWA=true"]