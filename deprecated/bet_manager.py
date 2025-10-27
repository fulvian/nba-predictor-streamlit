#!/usr/bin/env python3
"""
🎯 NBA Bet Manager - Gestione Scommesse Salvate
Visualizza, modifica e gestisce le scommesse pendenti
"""

import json
import os
from datetime import datetime
import argparse

class BetManager:
    def __init__(self):
        self.pending_file = 'data/pending_bets.json'
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """Assicura che la directory data esista."""
        os.makedirs('data', exist_ok=True)
    
    def load_pending_bets(self):
        """Carica le scommesse pendenti dal file JSON."""
        try:
            if not os.path.exists(self.pending_file):
                return []
            
            with open(self.pending_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_pending_bets(self, bets):
        """Salva le scommesse pendenti nel file JSON."""
        try:
            with open(self.pending_file, 'w') as f:
                json.dump(bets, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Errore nel salvataggio: {e}")
            return False
    
    def display_all_bets(self):
        """Visualizza tutte le scommesse salvate."""
        bets = self.load_pending_bets()
        
        if not bets:
            print("📝 Nessuna scommessa salvata trovata")
            return
        
        print(f"\n🎯 SCOMMESSE SALVATE ({len(bets)} totali)")
        print("=" * 80)
        
        # Raggruppa per status
        pending_bets = [b for b in bets if b.get('status') == 'pending']
        completed_bets = [b for b in bets if b.get('status') == 'completed']
        
        if pending_bets:
            print(f"\n⏳ SCOMMESSE PENDENTI ({len(pending_bets)})")
            print("-" * 50)
            for i, bet in enumerate(pending_bets, 1):
                self._print_bet_details(bet, i)
        
        if completed_bets:
            print(f"\n✅ SCOMMESSE COMPLETATE ({len(completed_bets)})")
            print("-" * 50)
            for i, bet in enumerate(completed_bets, 1):
                self._print_bet_details(bet, i, completed=True)
    
    def _print_bet_details(self, bet, index, completed=False):
        """Stampa i dettagli di una singola scommessa."""
        bet_data = bet.get('bet_data', {})
        game_id = bet.get('game_id', 'N/A')
        timestamp = bet.get('timestamp', 'N/A')
        
        # Parsing timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime('%d/%m/%Y %H:%M')
        except:
            formatted_time = timestamp
        
        print(f"\n{index}. 🎲 {bet_data.get('type', 'N/A')} {bet_data.get('line', 'N/A')} @ {bet_data.get('odds', 'N/A')}")
        print(f"   📅 Data: {formatted_time}")
        print(f"   🏀 Partita: {game_id}")
        print(f"   💰 Stake: €{bet_data.get('stake', 0):.2f}")
        print(f"   📊 Edge: {bet_data.get('edge', 0)*100:.1f}% | Prob: {bet_data.get('probability', 0)*100:.1f}%")
        
        if completed and 'result' in bet:
            result = bet['result']
            win_status = "🟢 VINTA" if result.get('bet_won') else "🔴 PERSA"
            profit_loss = result.get('profit_loss', 0)
            profit_symbol = "+" if profit_loss > 0 else ""
            print(f"   🎯 Risultato: {win_status} | P&L: {profit_symbol}€{profit_loss:.2f}")
            print(f"   📈 Totale Reale: {result.get('actual_total', 'N/A')}")
    
    def delete_bet(self, bet_index):
        """Elimina una scommessa specifica."""
        bets = self.load_pending_bets()
        
        if not bets or bet_index < 1 or bet_index > len(bets):
            print("❌ Indice scommessa non valido")
            return False
        
        bet_to_delete = bets[bet_index - 1]
        bet_data = bet_to_delete.get('bet_data', {})
        
        print(f"\n🗑️  ELIMINA SCOMMESSA:")
        print(f"   {bet_data.get('type', 'N/A')} {bet_data.get('line', 'N/A')} @ {bet_data.get('odds', 'N/A')}")
        print(f"   Stake: €{bet_data.get('stake', 0):.2f}")
        
        confirm = input("\nSei sicuro di voler eliminare questa scommessa? (y/N): ").strip().lower()
        if confirm == 'y':
            del bets[bet_index - 1]
            if self.save_pending_bets(bets):
                print("✅ Scommessa eliminata con successo!")
                return True
            else:
                print("❌ Errore nell'eliminazione")
                return False
        else:
            print("❌ Eliminazione annullata")
            return False
    
    def replace_bet(self, bet_index, new_bet_data, game_id):
        """Sostituisce una scommessa esistente con una nuova."""
        bets = self.load_pending_bets()
        
        if not bets or bet_index < 1 or bet_index > len(bets):
            print("❌ Indice scommessa non valido")
            return False
        
        old_bet = bets[bet_index - 1]
        old_bet_data = old_bet.get('bet_data', {})
        
        print(f"\n🔄 SOSTITUISCI SCOMMESSA:")
        print(f"   VECCHIA: {old_bet_data.get('type', 'N/A')} {old_bet_data.get('line', 'N/A')} @ {old_bet_data.get('odds', 'N/A')} (€{old_bet_data.get('stake', 0):.2f})")
        print(f"   NUOVA:   {new_bet_data.get('type', 'N/A')} {new_bet_data.get('line', 'N/A')} @ {new_bet_data.get('odds', 'N/A')} (€{new_bet_data.get('stake', 0):.2f})")
        
        confirm = input("\nConfermi la sostituzione? (y/N): ").strip().lower()
        if confirm == 'y':
            # Aggiorna la scommessa esistente
            bets[bet_index - 1] = {
                'bet_id': f"{game_id}_{new_bet_data['type']}_{new_bet_data['line']}",
                'game_id': game_id,
                'bet_data': new_bet_data,
                'timestamp': datetime.now().isoformat(),
                'status': 'pending',
                'replaced_at': datetime.now().isoformat(),
                'original_bet': old_bet_data
            }
            
            if self.save_pending_bets(bets):
                print("✅ Scommessa sostituita con successo!")
                return True
            else:
                print("❌ Errore nella sostituzione")
                return False
        else:
            print("❌ Sostituzione annullata")
            return False
    
    def clear_completed_bets(self):
        """Rimuove tutte le scommesse completate."""
        bets = self.load_pending_bets()
        
        completed_count = len([b for b in bets if b.get('status') == 'completed'])
        if completed_count == 0:
            print("📝 Nessuna scommessa completata da rimuovere")
            return
        
        print(f"\n🧹 PULIZIA SCOMMESSE COMPLETATE")
        print(f"   Trovate {completed_count} scommesse completate")
        
        confirm = input("Vuoi rimuoverle tutte? (y/N): ").strip().lower()
        if confirm == 'y':
            pending_only = [b for b in bets if b.get('status') == 'pending']
            if self.save_pending_bets(pending_only):
                print(f"✅ Rimosse {completed_count} scommesse completate!")
            else:
                print("❌ Errore nella pulizia")
        else:
            print("❌ Pulizia annullata")
    
    def get_bet_by_game_id(self, game_id):
        """Trova scommesse per un game_id specifico."""
        bets = self.load_pending_bets()
        matching_bets = []
        
        for i, bet in enumerate(bets):
            if bet.get('game_id') == game_id and bet.get('status') == 'pending':
                matching_bets.append((i + 1, bet))
        
        return matching_bets
    
    def check_bankroll(self):
        """Controlla e mostra lo stato del bankroll."""
        try:
            bankroll_file = 'data/bankroll.json'
            if not os.path.exists(bankroll_file):
                print("📝 File bankroll non trovato - Bankroll iniziale: €100.00")
                return
            
            with open(bankroll_file, 'r') as f:
                bankroll_data = json.load(f)
            
            current_bankroll = bankroll_data.get('current_bankroll', 100.0)
            
            print(f"\n💰 STATO BANKROLL")
            print("=" * 50)
            print(f"💵 Bankroll Attuale: €{current_bankroll:.2f}")
            
            # Calcola statistiche dalle scommesse
            bets = self.load_pending_bets()
            if bets:
                pending_bets = [b for b in bets if b.get('status') == 'pending']
                completed_bets = [b for b in bets if b.get('status') == 'completed']
                
                total_pending_stake = sum(b.get('bet_data', {}).get('stake', 0) for b in pending_bets)
                
                if completed_bets:
                    total_profit_loss = sum(b.get('result', {}).get('profit_loss', 0) for b in completed_bets)
                    wins = len([b for b in completed_bets if b.get('result', {}).get('bet_won', False)])
                    losses = len(completed_bets) - wins
                    win_rate = (wins / len(completed_bets)) * 100 if completed_bets else 0
                    
                    print(f"📊 Scommesse Completate: {len(completed_bets)} (🟢 {wins} vinte, 🔴 {losses} perse)")
                    print(f"📈 Win Rate: {win_rate:.1f}%")
                    print(f"💹 Profitto/Perdita Totale: {'+' if total_profit_loss > 0 else ''}€{total_profit_loss:.2f}")
                
                if pending_bets:
                    print(f"⏳ Scommesse Pendenti: {len(pending_bets)}")
                    print(f"💸 Stake Totale in Gioco: €{total_pending_stake:.2f}")
                    print(f"💰 Bankroll Disponibile: €{current_bankroll - total_pending_stake:.2f}")
                
                print(f"\n🎯 Esposizione Totale: €{total_pending_stake:.2f} ({(total_pending_stake/current_bankroll)*100:.1f}% del bankroll)")
            else:
                print("📋 Nessuna scommessa registrata")
                print("💰 Bankroll Disponibile: €{:.2f}".format(current_bankroll))
            
        except Exception as e:
            print(f"❌ Errore nel controllo bankroll: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='🎯 NBA Bet Manager - Sistema Completo di Gestione Scommesse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 ESEMPI D'USO:
     python bet_manager.py                                    # Visualizza tutte le scommesse
     python bet_manager.py --view                            # Stesso comando sopra
     python bet_manager.py --bankroll                        # Mostra stato bankroll dettagliato
     python bet_manager.py --delete 2                        # Elimina la scommessa numero 2
     python bet_manager.py --clean                           # Rimuovi tutte le scommesse completate
     python bet_manager.py --game-id "CUSTOM_Thunder_Pacers" # Cerca scommesse per partita specifica

STRUTTURA OUTPUT:
    ⏳ SCOMMESSE PENDENTI    - Scommesse in attesa di risultato
    ✅ SCOMMESSE COMPLETATE - Scommesse con risultato finale

CODICI COLORE:
    🟢 VINTA    - Scommessa vincente
    🔴 PERSA    - Scommessa perdente
    ⏳ PENDING  - In attesa di risultato
        """
    )
    
    parser.add_argument('--view', 
                       action='store_true', 
                       help='📊 Visualizza tutte le scommesse salvate (pendenti + completate)')
    
    parser.add_argument('--delete', 
                       type=int, 
                       metavar='NUMERO',
                       help='🗑️  Elimina scommessa specifica per numero (es: --delete 2)')
    
    parser.add_argument('--clean', 
                       action='store_true', 
                       help='🧹 Rimuovi tutte le scommesse completate (mantiene solo quelle pendenti)')
    
    parser.add_argument('--game-id', 
                       type=str, 
                       metavar='ID_PARTITA',
                       help='🔍 Cerca scommesse per una partita specifica (es: --game-id "CUSTOM_Thunder_Pacers")')
    
    parser.add_argument('--bankroll', 
                       action='store_true', 
                       help='💰 Mostra stato dettagliato del bankroll con statistiche')
    
    args = parser.parse_args()
    
    manager = BetManager()
    
    if args.view or (not any(vars(args).values())):
        manager.display_all_bets()
    
    if args.delete:
        manager.delete_bet(args.delete)
    
    if args.clean:
        manager.clear_completed_bets()
    
    if args.game_id:
        matching_bets = manager.get_bet_by_game_id(args.game_id)
        if matching_bets:
            print(f"\n🔍 SCOMMESSE PER GAME_ID: {args.game_id}")
            print("-" * 50)
            for index, bet in matching_bets:
                manager._print_bet_details(bet, index)
        else:
            print(f"📝 Nessuna scommessa pendente trovata per game_id: {args.game_id}")
    
    if args.bankroll:
        manager.check_bankroll()

if __name__ == "__main__":
    main() 