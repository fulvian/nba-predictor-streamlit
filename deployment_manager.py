# deployment_manager.py
"""
Sistema completo di deployment, ottimizzazioni performance e manutenzione
per il sistema momentum avanzato NBA.
"""

import os
import sys
import json
import pickle
import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional
import hashlib

class AdvancedCacheManager:
    """
    Sistema di cache avanzato per ottimizzare performance API calls.
    """
    
    def __init__(self, cache_dir='cache', max_cache_size_mb=500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        
        # Cache levels con TTL differenziati
        self.cache_configs = {
            'player_game_logs': {'ttl_hours': 2, 'priority': 'high'},     # Dati recenti, alta priorità
            'team_stats': {'ttl_hours': 6, 'priority': 'medium'},         # Stats squadra, media priorità  
            'season_data': {'ttl_hours': 24, 'priority': 'low'},          # Dati stagionali, bassa priorità
            'roster_data': {'ttl_hours': 12, 'priority': 'medium'},       # Roster cambia meno frequentemente
            'injury_reports': {'ttl_hours': 1, 'priority': 'critical'}    # Infortuni cambiano rapidamente
        }
        
        self.hit_stats = {cache_type: {'hits': 0, 'misses': 0} for cache_type in self.cache_configs}
        
    def _get_cache_key(self, cache_type: str, params: Dict) -> str:
        """Genera chiave cache univoca dai parametri."""
        param_str = json.dumps(params, sort_keys=True)
        return f"{cache_type}_{hashlib.md5(param_str.encode()).hexdigest()}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Restituisce path completo del file cache."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, cache_type: str, params: Dict) -> Optional[any]:
        """Recupera dati dalla cache se validi."""
        cache_key = self._get_cache_key(cache_type, params)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            self.hit_stats[cache_type]['misses'] += 1
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Controlla TTL
            cached_time = cache_data['timestamp']
            ttl_hours = self.cache_configs[cache_type]['ttl_hours']
            
            if datetime.now() - cached_time > timedelta(hours=ttl_hours):
                cache_path.unlink()  # Rimuovi cache scaduta
                self.hit_stats[cache_type]['misses'] += 1
                return None
            
            self.hit_stats[cache_type]['hits'] += 1
            return cache_data['data']
            
        except Exception as e:
            print(f"⚠️ Errore lettura cache {cache_key}: {e}")
            self.hit_stats[cache_type]['misses'] += 1
            return None
    
    def set(self, cache_type: str, params: Dict, data: any) -> None:
        """Salva dati in cache."""
        cache_key = self._get_cache_key(cache_type, params)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            'timestamp': datetime.now(),
            'data': data,
            'cache_type': cache_type,
            'params': params
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Controlla dimensione cache e pulisci se necessario
            self._cleanup_if_needed()
            
        except Exception as e:
            print(f"⚠️ Errore scrittura cache {cache_key}: {e}")
    
    def _cleanup_if_needed(self):
        """Pulisce cache se supera dimensione massima."""
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.pkl'))
        
        if cache_size > self.max_cache_size:
            # Ordina file per data modifica e priorità
            cache_files = []
            for f in self.cache_dir.glob('*.pkl'):
                try:
                    with open(f, 'rb') as file:
                        cache_data = pickle.load(file)
                    priority = self.cache_configs.get(cache_data['cache_type'], {}).get('priority', 'low')
                    cache_files.append((f, f.stat().st_mtime, priority))
                except:
                    cache_files.append((f, f.stat().st_mtime, 'low'))
            
            # Priorità: critical > high > medium > low, poi per data
            priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            cache_files.sort(key=lambda x: (priority_order.get(x[2], 1), x[1]))
            
            # Rimuovi primi file (meno prioritari e più vecchi)
            removed_size = 0
            target_reduction = cache_size - (self.max_cache_size * 0.8)  # Riduci a 80% max
            
            for file_path, _, _ in cache_files:
                if removed_size >= target_reduction:
                    break
                removed_size += file_path.stat().st_size
                file_path.unlink()
            
            print(f"🧹 Cache cleanup: rimossi {removed_size / 1024 / 1024:.1f}MB")
    
    def get_stats(self) -> Dict:
        """Restituisce statistiche cache."""
        total_hits = sum(stats['hits'] for stats in self.hit_stats.values())
        total_requests = sum(stats['hits'] + stats['misses'] for stats in self.hit_stats.values())
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.pkl'))
        
        return {
            'hit_rate_pct': round(hit_rate, 1),
            'total_requests': total_requests,
            'cache_size_mb': round(cache_size / 1024 / 1024, 1),
            'by_type': {
                cache_type: {
                    'hit_rate': round(stats['hits'] / max(stats['hits'] + stats['misses'], 1) * 100, 1),
                    'requests': stats['hits'] + stats['misses']
                }
                for cache_type, stats in self.hit_stats.items()
            }
        }

class OptimizedDataProvider:
    """
    Wrapper ottimizzato per NBADataProvider con caching avanzato.
    """
    
    def __init__(self, original_provider, cache_manager):
        self.original_provider = original_provider
        self.cache = cache_manager
        
        # Thread pool per richieste async
        self.request_queue = []
        self.request_lock = threading.Lock()
        
    def get_player_game_logs_cached(self, player_id, last_n_games=10):
        """Versione cached di get_player_game_logs."""
        cache_params = {
            'player_id': player_id,
            'last_n_games': last_n_games,
            'method': 'get_player_game_logs'
        }
        
        # Prova cache prima
        cached_data = self.cache.get('player_game_logs', cache_params)
        if cached_data is not None:
            return cached_data
        
        # Cache miss - recupera da API
        try:
            data = self.original_provider.get_player_game_logs(player_id, last_n_games)
            if data is not None:
                self.cache.set('player_game_logs', cache_params, data)
            return data
        except Exception as e:
            print(f"⚠️ Errore API player logs per {player_id}: {e}")
            return None
    
    def get_team_stats_cached(self, team_name, is_home=True):
        """Versione cached di _get_team_stats."""
        cache_params = {
            'team_name': team_name,
            'is_home': is_home,
            'method': 'get_team_stats'
        }
        
        cached_data = self.cache.get('team_stats', cache_params)
        if cached_data is not None:
            return cached_data
        
        try:
            data = self.original_provider._get_team_stats(team_name, is_home)
            if data and data.get('has_data'):
                self.cache.set('team_stats', cache_params, data)
            return data
        except Exception as e:
            print(f"⚠️ Errore API team stats per {team_name}: {e}")
            return None
    
    def get_team_roster_cached(self, team_id, season=None):
        """Versione cached di get_team_roster."""
        cache_params = {
            'team_id': team_id,
            'season': season or 'current',
            'method': 'get_team_roster'
        }
        
        cached_data = self.cache.get('roster_data', cache_params)
        if cached_data is not None:
            return cached_data
        
        try:
            data = self.original_provider.get_team_roster(team_id, season)
            if data:
                self.cache.set('roster_data', cache_params, data)
            return data
        except Exception as e:
            print(f"⚠️ Errore API roster per team {team_id}: {e}")
            return []
    
    def prefetch_common_data(self, teams_to_analyze: List[str]):
        """
        Pre-carica dati comuni per ridurre latenza durante analisi.
        """
        print("🚀 Pre-caricamento dati comuni...")
        
        def prefetch_worker(team_name):
            try:
                # Pre-carica stats squadra
                self.get_team_stats_cached(team_name, True)
                self.get_team_stats_cached(team_name, False)
                
                # Pre-carica roster se abbiamo team_id
                team_info = self.original_provider._find_team_by_name(team_name)
                if team_info:
                    self.get_team_roster_cached(team_info['id'])
                    
            except Exception as e:
                print(f"   ⚠️ Errore prefetch per {team_name}: {e}")
        
        # Esegui prefetch in parallelo
        threads = []
        for team in teams_to_analyze:
            thread = threading.Thread(target=prefetch_worker, args=(team,))
            threads.append(thread)
            thread.start()
        
        # Attendi completamento
        for thread in threads:
            thread.join()
        
        print("✅ Pre-caricamento completato")

class SystemMaintenanceManager:
    """
    Gestisce manutenzione automatica del sistema.
    """
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.backup_dir = self.base_dir / 'backups'
        self.backup_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.base_dir / 'logs'
        self.models_dir = self.base_dir / 'models'
        self.cache_dir = self.base_dir / 'cache'
        
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Crea backup completo del sistema."""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        print(f"📦 Creando backup: {backup_name}")
        
        # Backup modelli
        if self.models_dir.exists():
            shutil.copytree(self.models_dir, backup_path / 'models', dirs_exist_ok=True)
        
        # Backup database monitoring
        db_files = list(self.base_dir.glob('*.db'))
        for db_file in db_files:
            shutil.copy2(db_file, backup_path)
        
        # Backup configurazioni
        config_files = ['config.py', 'requirements.txt', 'settings.json']
        for config_file in config_files:
            config_path = self.base_dir / config_file
            if config_path.exists():
                shutil.copy2(config_path, backup_path)
        
        # Backup logs recenti (ultimi 7 giorni)
        if self.log_dir.exists():
            log_backup_dir = backup_path / 'logs'
            log_backup_dir.mkdir(exist_ok=True)
            
            cutoff_date = datetime.now() - timedelta(days=7)
            for log_file in self.log_dir.glob('*.log'):
                if datetime.fromtimestamp(log_file.stat().st_mtime) > cutoff_date:
                    shutil.copy2(log_file, log_backup_dir)
        
        # Crea manifest del backup
        manifest = {
            'backup_name': backup_name,
            'created_at': datetime.now().isoformat(),
            'files_included': [str(f.relative_to(backup_path)) for f in backup_path.rglob('*') if f.is_file()],
            'system_version': self._get_system_version()
        }
        
        with open(backup_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Comprimi backup
        archive_path = f"{backup_path}.tar.gz"
        shutil.make_archive(str(backup_path), 'gztar', str(backup_path))
        shutil.rmtree(backup_path)  # Rimuovi directory temporanea
        
        print(f"✅ Backup creato: {archive_path}")
        return archive_path
    
    def cleanup_old_logs(self, keep_days=30):
        """Pulisce log più vecchi di N giorni."""
        if not self.log_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        removed_count = 0
        
        for log_file in self.log_dir.glob('*.log'):
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                log_file.unlink()
                removed_count += 1
        
        print(f"🧹 Rimossi {removed_count} file di log vecchi")
    
    def cleanup_old_backups(self, keep_count=10):
        """Mantiene solo gli ultimi N backup."""
        backups = sorted(self.backup_dir.glob('backup_*.tar.gz'), 
                        key=lambda x: x.stat().st_mtime, reverse=True)
        
        removed_count = 0
        for backup in backups[keep_count:]:
            backup.unlink()
            removed_count += 1
        
        if removed_count > 0:
            print(f"🧹 Rimossi {removed_count} backup vecchi")
    
    def check_system_health(self) -> Dict:
        """Controlla salute generale del sistema."""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Controlla spazio disco
        disk_usage = shutil.disk_usage(self.base_dir)
        free_space_gb = disk_usage.free / (1024**3)
        
        if free_space_gb < 1.0:  # Meno di 1GB libero
            health_report['issues'].append("Spazio disco insufficiente")
            health_report['recommendations'].append("Pulire cache e log")
            health_report['status'] = 'warning'
        
        # Controlla dimensione cache
        if self.cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            cache_size_mb = cache_size / (1024**2)
            
            if cache_size_mb > 1000:  # Cache > 1GB
                health_report['issues'].append(f"Cache troppo grande: {cache_size_mb:.1f}MB")
                health_report['recommendations'].append("Ridurre dimensione cache massima")
        
        # Controlla file modelli
        critical_files = ['mu_model.pkl', 'sigma_model.pkl', 'scaler.pkl']
        missing_models = []
        
        for model_file in critical_files:
            if not (self.models_dir / 'probabilistic' / model_file).exists():
                missing_models.append(model_file)
        
        if missing_models:
            health_report['issues'].append(f"Modelli mancanti: {missing_models}")
            health_report['recommendations'].append("Ri-addestrare o ripristinare modelli")
            health_report['status'] = 'critical'
        
        # Controlla log errors recenti
        if self.log_dir.exists():
            recent_errors = self._count_recent_errors()
            if recent_errors > 10:  # Più di 10 errori nelle ultime 24h
                health_report['issues'].append(f"Molti errori recenti: {recent_errors}")
                health_report['recommendations'].append("Investigare log errors")
                health_report['status'] = 'warning'
        
        return health_report
    
    def _get_system_version(self) -> str:
        """Determina versione sistema da git o file."""
        try:
            # Prova git
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode == 0:
                return f"git-{result.stdout.strip()}"
        except:
            pass
        
        # Fallback a timestamp
        return f"dev-{datetime.now().strftime('%Y%m%d')}"
    
    def _count_recent_errors(self) -> int:
        """Conta errori nei log delle ultime 24h."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        error_count = 0
        
        for log_file in self.log_dir.glob('*.log'):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if 'ERROR' in line or '❌' in line:
                            # Estrai timestamp dal log (assumendo formato standard)
                            try:
                                log_time_str = line.split('|')[0].strip()
                                log_time = datetime.strptime(log_time_str, '%H:%M:%S')
                                # Aggiorna con data corrente
                                log_time = log_time.replace(year=datetime.now().year,
                                                          month=datetime.now().month,
                                                          day=datetime.now().day)
                                if log_time > cutoff_time:
                                    error_count += 1
                            except:
                                # Se non riusciamo a parsare il timestamp, conta comunque l'errore
                                error_count += 1
            except:
                continue
        
        return error_count
    
    def run_daily_maintenance(self):
        """Esegue manutenzione giornaliera automatica."""
        print("🔧 Avvio manutenzione giornaliera...")
        
        # Backup giornaliero
        self.create_backup()
        
        # Pulizia
        self.cleanup_old_logs(keep_days=30)
        self.cleanup_old_backups(keep_count=7)
        
        # Health check
        health = self.check_system_health()
        
        if health['status'] != 'healthy':
            print(f"⚠️ Sistema non healthy: {health['status']}")
            for issue in health['issues']:
                print(f"   - {issue}")
        else:
            print("✅ Sistema healthy")
        
        return health

class DeploymentManager:
    """
    Gestisce deployment e aggiornamenti del sistema.
    """
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.maintenance = SystemMaintenanceManager(base_dir)
        self.cache_manager = AdvancedCacheManager()
        
    def deploy_momentum_system(self, config_profile='balanced'):
        """
        Deployment completo del sistema momentum avanzato.
        """
        print("🚀 Avvio deployment sistema momentum avanzato...")
        
        # 1. Verifica prerequisiti
        if not self._check_prerequisites():
            print("❌ Prerequisiti non soddisfatti")
            return False
        
        # 2. Backup sistema esistente
        backup_path = self.maintenance.create_backup('pre_deployment')
        print(f"📦 Backup pre-deployment: {backup_path}")
        
        # 3. Setup cache ottimizzato
        print("⚡ Configurando cache ottimizzato...")
        
        # 4. Configura profilo
        from advanced_momentum_config import AdvancedMomentumConfig
        config = AdvancedMomentumConfig(config_profile)
        print(f"⚙️ Applicando profilo: {config.config['name']}")
        
        # 5. Test sistema
        if not self._run_deployment_tests():
            print("❌ Test deployment falliti")
            return False
        
        # 6. Setup monitoring
        print("📊 Configurando monitoring...")
        
        # 7. Setup manutenzione automatica
        self._setup_automated_maintenance()
        
        print("✅ Deployment completato con successo!")
        return True
    
    def _check_prerequisites(self) -> bool:
        """Verifica prerequisiti per deployment."""
        print("🔍 Verifica prerequisiti...")
        
        required_files = [
            'advanced_player_momentum_predictor.py',
            'data_provider.py',
            'main.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.base_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ File mancanti: {missing_files}")
            return False
        
        # Verifica dipendenze Python
        required_packages = ['pandas', 'numpy', 'scipy', 'scikit-learn']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Pacchetti Python mancanti: {missing_packages}")
            print("   Installa con: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ Prerequisiti soddisfatti")
        return True
    
    def _run_deployment_tests(self) -> bool:
        """Esegue test di deployment."""
        print("🧪 Esecuzione test deployment...")
        
        try:
            # Test import moduli
            sys.path.insert(0, str(self.base_dir))
            
            from advanced_player_momentum_predictor import AdvancedPlayerMomentumPredictor
            from data_provider import NBADataProvider
            
            # Test inizializzazione componenti
            data_provider = NBADataProvider()
            momentum_predictor = AdvancedPlayerMomentumPredictor(nba_data_provider=data_provider)
            
            # Test configurazione cache
            optimized_provider = OptimizedDataProvider(data_provider, self.cache_manager)
            
            print("✅ Test deployment superati")
            return True
            
        except Exception as e:
            print(f"❌ Test deployment fallito: {e}")
            return False
    
    def _setup_automated_maintenance(self):
        """Configura manutenzione automatica."""
        print("🔄 Configurando manutenzione automatica...")
        
        # Crea script di manutenzione giornaliera
        maintenance_script = f'''#!/usr/bin/env python3
"""Script di manutenzione automatica generato"""

import sys
sys.path.append('{self.base_dir}')

from deployment_manager import SystemMaintenanceManager

if __name__ == "__main__":
    maintenance = SystemMaintenanceManager('{self.base_dir}')
    health = maintenance.run_daily_maintenance()
    
    if health['status'] == 'critical':
        sys.exit(1)
    else:
        sys.exit(0)
'''
        
        script_path = self.base_dir / 'daily_maintenance.py'
        with open(script_path, 'w') as f:
            f.write(maintenance_script)
        
        # Rendi eseguibile (Unix)
        try:
            script_path.chmod(0o755)
        except:
            pass
        
        print(f"📝 Script manutenzione creato: {script_path}")
        print("   Configura cron job per esecuzione giornaliera:")
        print(f"   0 2 * * * {sys.executable} {script_path}")

# Script principale di deployment
def main():
    """Script principale di deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema di deployment momentum NBA")
    parser.add_argument('--profile', default='balanced', 
                       choices=['conservative', 'balanced', 'aggressive', 'research_optimal'],
                       help='Profilo di configurazione')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Salta test di deployment')
    parser.add_argument('--maintenance-only', action='store_true',
                       help='Esegui solo manutenzione')
    
    args = parser.parse_args()
    
    deployment_manager = DeploymentManager()
    
    if args.maintenance_only:
        health = deployment_manager.maintenance.run_daily_maintenance()
        print(f"\n📊 Health status: {health['status']}")
        if health['issues']:
            print("Issues:")
            for issue in health['issues']:
                print(f"   - {issue}")
        return
    
    # Deployment completo
    success = deployment_manager.deploy_momentum_system(args.profile)
    
    if success:
        print("\n🎉 Sistema momentum avanzato deployato con successo!")
        print("\nProssimi passi:")
        print("1. Testa il sistema con: python main.py --giorni 1")
        print("2. Monitora performance nei prossimi giorni")
        print("3. Aggiusta configurazione se necessario")
        print("4. Setup cron job per manutenzione automatica")
    else:
        print("\n❌ Deployment fallito. Controlla log per dettagli.")

if __name__ == "__main__":
    main()