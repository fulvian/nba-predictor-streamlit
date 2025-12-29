#!/bin/bash
echo "🔐 Generating Betfair API Certificates..."

# 1. Create directory
mkdir -p certs
cd certs

# 2. Generate Private Key (client-2048.key) - No password for automation (or secure handling)
# Betfair docs suggest unencrypted key for simple bots, but best practice is encrypted. 
# For now, we'll generate an unencrypted key to avoid password prompts during bot run.
openssl genrsa -out client-2048.key 2048

# 3. Generate Certificate Signing Request (client-2048.csr)
# Automated subject line to avoid interactive prompts
openssl req -new -config ../openssl.cnf -key client-2048.key -out client-2048.csr -subj "/C=IT/ST=Italy/L=Rome/O=NBAPredictor/CN=nba-live-bot"

# 4. Generate Self-Signed Certificate (client-2048.crt)
openssl x509 -req -days 365 -in client-2048.csr -signkey client-2048.key -out client-2048.crt

echo "✅ Certificates generated in /certs/:"
echo "   - client-2048.key (Private Key - Keep Secure!)"
echo "   - client-2048.crt (Public Certificate - UPLOAD TO BETFAIR)"
