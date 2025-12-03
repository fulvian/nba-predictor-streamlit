#!/bin/bash

# NBA Predictor - Production Deployment Script
# Context7 SuperPoteri Deployment Automation
# Version: 5.0.0

set -euo pipefail

# ===========================================
# Configuration
# ===========================================
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly DEPLOYMENT_LOG="$PROJECT_ROOT/logs/deployment_$(date +%Y%m%d_%H%M%S).log"
readonly BACKUP_DIR="$PROJECT_ROOT/backups/deployment_$(date +%Y%m%d_%H%M%S)"
readonly COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"

# ===========================================
# Colors for Output
# ===========================================
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# ===========================================
# Logging Functions
# ===========================================
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

# ===========================================
# Utility Functions
# ===========================================
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if running as root (not recommended for production)
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root is not recommended for production deployments"
    fi
    
    # Check if .env.production exists
    if [[ ! -f "$PROJECT_ROOT/.env.production" ]]; then
        log_error ".env.production file not found. Please create it first."
        exit 1
    fi
    
    # Check required directories
    local required_dirs=("logs" "backups" "secrets" "ssl" "nginx" "monitoring")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$PROJECT_ROOT/$dir" ]]; then
            log_warning "Directory $dir does not exist. Creating it..."
            mkdir -p "$PROJECT_ROOT/$dir"
        fi
    done
    
    log_success "Prerequisites check completed"
}

backup_current_deployment() {
    log "Creating backup of current deployment..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup Docker volumes
    if docker volume ls | grep -q "nba"; then
        log "Backing up Docker volumes..."
        docker run --rm -v nba_cache-data:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/cache_data.tar.gz -C /data .
    fi
    
    # Backup configuration files
    log "Backing up configuration files..."
    cp -r "$PROJECT_ROOT/config" "$BACKUP_DIR/" 2>/dev/null || true
    cp -r "$PROJECT_ROOT/secrets" "$BACKUP_DIR/" 2>/dev/null || true
    cp "$PROJECT_ROOT/.env.production" "$BACKUP_DIR/" 2>/dev/null || true
    
    # Backup databases
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        log "Backing up databases..."
        cp -r "$PROJECT_ROOT/data" "$BACKUP_DIR/"
    fi
    
    log_success "Backup created at $BACKUP_DIR"
}

validate_configuration() {
    log "Validating deployment configuration..."
    
    # Validate Docker Compose file
    if ! docker-compose -f "$COMPOSE_FILE" config > /dev/null 2>&1; then
        log_error "Docker Compose configuration is invalid"
        docker-compose -f "$COMPOSE_FILE" config
        exit 1
    fi
    
    # Check if all required secrets exist
    local required_secrets=("nba_api_secret.txt" "redis_password.txt")
    for secret in "${required_secrets[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/secrets/$secret" ]]; then
            log_error "Required secret file $secret not found in secrets directory"
            exit 1
        fi
    done
    
    # Validate SSL certificates
    if [[ ! -f "$PROJECT_ROOT/ssl/cert.pem" ]] || [[ ! -f "$PROJECT_ROOT/ssl/key.pem" ]]; then
        log_warning "SSL certificates not found. HTTP only mode will be used."
    fi
    
    log_success "Configuration validation completed"
}

build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    log "Building NBA Predictor application image..."
    docker build -t nba-predictor:5.0.0 --target production .
    
    # Build WebSocket image if Dockerfile.websocket exists
    if [[ -f "Dockerfile.websocket" ]]; then
        log "Building WebSocket service image..."
        docker build -f Dockerfile.websocket -t nba-websocket:5.0.0 --target production .
    fi
    
    log_success "Docker images built successfully"
}

deploy_services() {
    log "Deploying NBA Predictor services..."
    
    cd "$PROJECT_ROOT"
    
    # Load environment variables
    set -a
    source .env.production
    set +a
    
    # Deploy services
    docker-compose -f "$COMPOSE_FILE" down
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log_success "Services deployed successfully"
}

wait_for_services() {
    log "Waiting for services to be ready..."
    
    local services=("nba-predictor-api:8501" "nba-predictor-dashboard:8502" "context7-cache:6379")
    local max_wait_time=300
    local wait_interval=10
    
    for service in "${services[@]}"; do
        local service_name=$(echo "$service" | cut -d':' -f1)
        local port=$(echo "$service" | cut -d':' -f2)
        local wait_time=0
        
        log "Waiting for $service_name to be ready..."
        
        while [[ $wait_time -lt $max_wait_time ]]; do
            if docker-compose -f "$COMPOSE_FILE" exec "$service_name" curl -f "http://localhost:$port/health" > /dev/null 2>&1; then
                log_success "$service_name is ready"
                break
            fi
            
            sleep $wait_interval
            wait_time=$((wait_time + wait_interval))
            
            if [[ $wait_time -ge $max_wait_time ]]; then
                log_error "$service_name failed to start within $max_wait_time seconds"
                return 1
            fi
        done
    done
    
    log_success "All services are ready"
}

run_health_checks() {
    log "Running comprehensive health checks..."
    
    # Check service containers
    local containers=("nba-predictor-api-prod" "nba-predictor-dashboard-prod" "context7-intelligent-cache-prod" "nginx-lb-prod")
    
    for container in "${containers[@]}"; do
        if docker ps --filter "name=$container" --filter "status=running" | grep -q "$container"; then
            log_success "Container $container is running"
        else
            log_error "Container $container is not running"
            return 1
        fi
    done
    
    # Check API endpoints
    log "Testing API endpoints..."
    if curl -f "http://localhost/health" > /dev/null 2>&1; then
        log_success "Load balancer health check passed"
    else
        log_error "Load balancer health check failed"
        return 1
    fi
    
    log_success "All health checks passed"
}

rollback_deployment() {
    log_warning "Initiating deployment rollback..."
    
    cd "$PROJECT_ROOT"
    
    # Stop current services
    docker-compose -f "$COMPOSE_FILE" down
    
    # Restore from backup if available
    if [[ -d "$BACKUP_DIR" ]] && [[ -d "$BACKUP_DIR/data" ]]; then
        log "Restoring data from backup..."
        rm -rf "$PROJECT_ROOT/data"
        cp -r "$BACKUP_DIR/data" "$PROJECT_ROOT/"
    fi
    
    # Restart services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log_success "Rollback completed"
}

cleanup() {
    log "Performing cleanup..."
    
    # Remove unused Docker images
    docker image prune -f > /dev/null 2>&1
    
    # Clean old logs (keep last 7 days)
    find "$PROJECT_ROOT/logs" -name "*.log" -mtime +7 -delete > /dev/null 2>&1
    
    # Clean old backups (keep last 30 days)
    find "$PROJECT_ROOT/backups" -type d -mtime +30 -exec rm -rf {} + > /dev/null 2>&1
    
    log_success "Cleanup completed"
}

# ===========================================
# Main Deployment Flow
# ===========================================
main() {
    log "Starting NBA Predictor production deployment..."
    log "Deployment version: 5.0.0"
    log "Context7 SuperPoteri enabled: true"
    
    # Trap for cleanup on exit
    trap 'log_error "Deployment failed. Check logs at $DEPLOYMENT_LOG"' ERR
    trap cleanup EXIT
    
    # Execute deployment steps
    check_prerequisites
    backup_current_deployment
    validate_configuration
    build_images
    deploy_services
    wait_for_services
    run_health_checks
    
    log_success "NBA Predictor deployment completed successfully!"
    log "Deployment log: $DEPLOYMENT_LOG"
    log "Backup location: $BACKUP_DIR"
    
    # Display service URLs
    echo
    log "Service URLs:"
    log "  Dashboard: https://localhost/"
    log "  API: https://localhost/api/"
    log "  Grafana: https://localhost:3000/"
    log "  Prometheus: https://localhost:9090/"
}

# ===========================================
# Script Entry Point
# ===========================================
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "health")
        run_health_checks
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health|cleanup}"
        echo "  deploy    - Deploy NBA Predictor to production"
        echo "  rollback  - Rollback to previous deployment"
        echo "  health    - Run health checks"
        echo "  cleanup   - Clean up unused resources"
        exit 1
        ;;
esac