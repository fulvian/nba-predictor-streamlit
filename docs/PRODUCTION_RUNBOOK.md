# NBA Predictor - Production Runbook
## Context7 SuperPoteri Deployment Guide

### Table of Contents
1. [System Overview](#system-overview)
2. [Deployment Procedures](#deployment-procedures)
3. [Monitoring and Alerting](#monitoring-and-alerting)
4. [Troubleshooting](#troubleshooting)
5. [Emergency Procedures](#emergency-procedures)
6. [Maintenance Tasks](#maintenance-tasks)
7. [Rollback Procedures](#rollback-procedures)

---

## System Overview

### Architecture Components
- **NBA Predictor API**: Core ML prediction service (Port 8501)
- **NBA Dashboard**: Streamlit web interface (Port 8502)
- **Context7 Cache**: Redis intelligent caching (Port 6379)
- **WebSocket Service**: Real-time updates (Port 8080)
- **Nginx Load Balancer**: SSL termination and routing (Ports 80/443)
- **Prometheus**: Metrics collection (Port 9090)
- **Grafana**: Visualization dashboard (Port 3000)

### Key Technologies
- **Docker & Docker Compose**: Container orchestration
- **Context7 SuperPoteri**: Advanced optimization framework
- **PWA**: Progressive Web App capabilities
- **ML Pipeline**: UnifiedHybridPipeline with ensemble models
- **Intelligent Caching**: Redis with TTL and performance monitoring

---

## Deployment Procedures

### Prerequisites
```bash
# Verify Docker installation
docker --version
docker-compose --version

# Check required directories
ls -la logs/ backups/ secrets/ ssl/ nginx/ monitoring/

# Verify environment file
ls -la .env.production
```

### Standard Deployment
```bash
# Execute deployment script
./scripts/deploy.sh deploy

# Monitor deployment progress
tail -f logs/deployment_*.log
```

### Deployment Steps
1. **Pre-deployment Checks**
   - Validate configuration files
   - Check service dependencies
   - Verify SSL certificates
   - Create backup of current deployment

2. **Build and Deploy**
   - Build Docker images
   - Deploy services with Docker Compose
   - Wait for service readiness
   - Run health checks

3. **Post-deployment Validation**
   - Run integration tests
   - Verify monitoring endpoints
   - Check application functionality
   - Validate Context7 features

### Environment Variables
Key production variables in `.env.production`:
```bash
# Application
NBA_PREDICTOR_ENV=production
CONTEXT7_ENABLED=true
SUPERPOTERI_MODE=active

# Database
DATABASE_URL=duckdb:///data/nba_production.duckdb
CACHE_TTL=3600

# Security
SSL_CERT_PATH=/app/ssl/cert.pem
SSL_KEY_PATH=/app/ssl/key.pem

# Monitoring
METRICS_ENABLED=true
LOG_LEVEL=INFO
```

---

## Monitoring and Alerting

### Key Metrics to Monitor

#### Application Metrics
- **API Response Time**: < 2 seconds (P95)
- **Dashboard Load Time**: < 3 seconds (P95)
- **Prediction Latency**: < 500ms (P95)
- **Error Rate**: < 1% (4xx/5xx responses)
- **Cache Hit Rate**: > 80%

#### Infrastructure Metrics
- **CPU Usage**: < 70% average
- **Memory Usage**: < 80% average
- **Disk Usage**: < 85% total
- **Network Latency**: < 100ms internal

#### Business Metrics
- **Prediction Accuracy**: > 65% (target)
- **Daily Active Users**: Track trends
- **API Requests per Minute**: Monitor load
- **WebSocket Connections**: Active sessions

### Monitoring Tools

#### Prometheus Metrics
Access: `https://your-domain.com:9090`

Key queries:
```promql
# API error rate
rate(http_requests_total{status=~"5.."}[5m])

# Cache hit rate
rate(redis_keyspace_hits_total[5m]) / (rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m]))

# Prediction latency
histogram_quantile(0.95, rate(ml_prediction_duration_seconds_bucket[5m]))
```

#### Grafana Dashboards
Access: `https://your-domain.com:3000`

Available dashboards:
- NBA Predictor Overview
- API Performance
- Cache Performance
- Infrastructure Health
- Context7 SuperPoteri Metrics

### Alert Configuration

#### Critical Alerts (Immediate)
- Service down (> 1 minute)
- Error rate > 5%
- Response time > 5 seconds
- Disk usage > 90%

#### Warning Alerts (Within 1 hour)
- Error rate > 2%
- Response time > 3 seconds
- Cache hit rate < 70%
- Memory usage > 85%

---

## Troubleshooting

### Common Issues and Solutions

#### Service Not Starting
**Symptoms**: Container exits immediately or fails health checks

**Diagnostics**:
```bash
# Check container logs
docker-compose -f docker-compose.prod.yml logs nba-predictor-api

# Check container status
docker ps -a

# Check resource usage
docker stats
```

**Solutions**:
1. Verify environment variables
2. Check port conflicts
3. Validate configuration files
4. Restart affected service

#### High Response Times
**Symptoms**: API responses > 2 seconds

**Diagnostics**:
```bash
# Check system resources
docker stats
top

# Check database performance
docker-compose exec nba-predictor-api python -c "
from src.nba_predictor.core.data_store import UnifiedDataStore
store = UnifiedDataStore()
print(store.get_performance_stats())
"

# Check cache performance
docker-compose exec context7-cache redis-cli info stats
```

**Solutions**:
1. Scale API service
2. Optimize database queries
3. Check cache configuration
4. Review ML pipeline performance

#### Cache Issues
**Symptoms**: Low hit rate, memory issues

**Diagnostics**:
```bash
# Check Redis stats
docker-compose exec context7-cache redis-cli info memory
docker-compose exec context7-cache redis-cli info stats

# Check cache configuration
docker-compose exec context7-cache redis-cli config get "*"
```

**Solutions**:
1. Adjust memory limits
2. Review eviction policies
3. Optimize TTL settings
4. Restart cache service

#### SSL Certificate Issues
**Symptoms**: HTTPS not working, certificate errors

**Diagnostics**:
```bash
# Check certificate validity
openssl x509 -in ssl/cert.pem -text -noout

# Check nginx configuration
docker-compose exec nginx-lb nginx -t

# Test SSL connection
openssl s_client -connect your-domain.com:443
```

**Solutions**:
1. Renew certificates
2. Update nginx configuration
3. Restart nginx service
4. Verify certificate chain

### Debug Commands

#### Application Debugging
```bash
# Enter container shell
docker-compose exec nba-predictor-api bash

# Check application logs
docker-compose logs -f nba-predictor-api

# Test API endpoints
curl -X GET "http://localhost:8501/health"
curl -X POST "http://localhost:8501/predict" -H "Content-Type: application/json" -d '{"home_team": "Lakers", "away_team": "Celtics"}'
```

#### Database Debugging
```bash
# Connect to database
docker-compose exec nba-predictor-api python -c "
from src.nba_predictor.core.data_store import UnifiedDataStore
store = UnifiedDataStore()
print('Database connection:', store.test_connection())
"

# Check database stats
docker-compose exec nba-predictor-api python -c "
from src.nba_predictor.core.data_store import UnifiedDataStore
store = UnifiedDataStore()
print(store.get_database_stats())
"
```

---

## Emergency Procedures

### Service Outage Response

#### Immediate Actions (First 5 minutes)
1. **Assess Impact**
   - Check monitoring dashboards
   - Verify service status
   - Identify affected components

2. **Communication**
   - Notify stakeholders
   - Update status page
   - Document incident start time

3. **Initial Triage**
   - Check recent deployments
   - Review error logs
   - Identify root cause category

#### Recovery Actions (First 30 minutes)
1. **Quick Fixes**
   - Restart affected services
   - Rollback recent changes if needed
   - Scale up healthy services

2. **Workarounds**
   - Enable maintenance mode if needed
   - Redirect traffic to healthy instances
   - Provide alternative access methods

3. **Monitoring**
   - Watch recovery progress
   - Validate service restoration
   - Continue monitoring for recurrence

### Data Recovery Procedures

#### Database Recovery
```bash
# Stop application services
docker-compose -f docker-compose.prod.yml down

# Restore from backup
cp backups/latest/nba_production.duckdb data/nba_production.duckdb

# Restart services
docker-compose -f docker-compose.prod.yml up -d

# Verify data integrity
./scripts/integration_test.sh run
```

#### Cache Recovery
```bash
# Clear corrupted cache
docker-compose exec context7-cache redis-cli FLUSHALL

# Restart cache service
docker-compose restart context7-cache

# Warm up cache
curl -X GET "http://localhost:8501/cache/warmup"
```

---

## Maintenance Tasks

### Daily Tasks
- [ ] Check system health dashboards
- [ ] Review error logs for anomalies
- [ ] Verify backup completion
- [ ] Monitor resource utilization

### Weekly Tasks
- [ ] Update security patches
- [ ] Review performance trends
- [ ] Clean up old logs and backups
- [ ] Test disaster recovery procedures

### Monthly Tasks
- [ ] SSL certificate renewal check
- [ ] Capacity planning review
- [ ] Security audit
- [ ] Performance optimization review

### Quarterly Tasks
- [ ] Major version updates
- [ ] Architecture review
- [ ] Disaster recovery testing
- [ ] Documentation updates

---

## Rollback Procedures

### Automated Rollback
```bash
# Execute rollback script
./scripts/deploy.sh rollback

# Monitor rollback progress
tail -f logs/deployment_*.log

# Verify system stability
./scripts/integration_test.sh run
```

### Manual Rollback Steps

#### 1. Identify Last Stable Version
```bash
# Check deployment history
ls -la logs/deployment_*.log

# Review previous successful deployment
grep "SUCCESS" logs/deployment_*.log | tail -5
```

#### 2. Restore Previous Configuration
```bash
# Stop current services
docker-compose -f docker-compose.prod.yml down

# Restore configuration
cp backups/deployment_YYYYMMDD_HHMMSS/.env.production .env.production
cp backups/deployment_YYYYMMDD_HHMMSS/docker-compose.prod.yml docker-compose.prod.yml

# Restart services
docker-compose -f docker-compose.prod.yml up -d
```

#### 3. Validate Rollback
```bash
# Run health checks
./scripts/deploy.sh health

# Run integration tests
./scripts/integration_test.sh run

# Monitor system stability
watch -n 5 'docker ps && docker stats --no-stream'
```

### Rollback Validation Checklist
- [ ] All services running
- [ ] Health checks passing
- [ ] API endpoints responding
- [ ] Dashboard accessible
- [ ] Cache connectivity working
- [ ] Monitoring endpoints active
- [ ] No error spikes in logs
- [ ] Performance metrics normal

---

## Contact Information

### Primary Contacts
- **DevOps Lead**: [Email/Phone]
- **Development Lead**: [Email/Phone]
- **System Administrator**: [Email/Phone]

### Escalation Contacts
- **CTO**: [Email/Phone]
- **Incident Manager**: [Email/Phone]
- **Security Team**: [Email/Phone]

### External Services
- **Cloud Provider**: Support Contact
- **SSL Certificate Provider**: Support Contact
- **Monitoring Service**: Support Contact

---

## Appendix

### Useful Commands
```bash
# Service status
docker-compose -f docker-compose.prod.yml ps

# Resource usage
docker stats --no-stream

# Log monitoring
docker-compose -f docker-compose.prod.yml logs -f

# Health checks
curl -f http://localhost/health

# Integration tests
./scripts/integration_test.sh run

# Performance monitoring
./scripts/performance_check.sh
```

### Configuration Files
- `docker-compose.prod.yml`: Production service definitions
- `.env.production`: Production environment variables
- `nginx/nginx.prod.conf`: Load balancer configuration
- `monitoring/prometheus.yml`: Metrics collection configuration
- `monitoring/grafana/`: Dashboard configurations

### Log Locations
- Application logs: `logs/`
- Deployment logs: `logs/deployment_*.log`
- Integration test logs: `logs/integration_test_*.log`
- Docker logs: `docker-compose logs`

### Backup Locations
- Database backups: `backups/`
- Configuration backups: `backups/deployment_*/`
- SSL certificates: `ssl/`
- Secrets: `secrets/`

---

**Last Updated**: $(date +%Y-%m-%d)
**Version**: 5.0.0
**Context7 SuperPoteri**: Enabled