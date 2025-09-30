#!/bin/bash
# Fix DNS resolution issue in WSL2

echo "Fixing DNS resolution issue..."

# Backup current resolv.conf
sudo cp /etc/resolv.conf /etc/resolv.conf.backup

# Create new resolv.conf with Google DNS
sudo bash -c 'cat > /etc/resolv.conf << EOF
# Google Public DNS
nameserver 8.8.8.8
nameserver 8.8.4.4
# Cloudflare DNS as backup
nameserver 1.1.1.1
EOF'

# Test DNS resolution
echo "Testing DNS resolution..."
if ping -c 1 github.com > /dev/null 2>&1; then
    echo "✅ DNS resolution fixed! GitHub is reachable."
else
    echo "❌ DNS still not working. Trying alternative method..."

    # Alternative: Add GitHub to hosts file
    sudo bash -c 'echo "140.82.114.4 github.com" >> /etc/hosts'
    sudo bash -c 'echo "140.82.114.3 api.github.com" >> /etc/hosts'
    echo "Added GitHub to hosts file as fallback"
fi

# Test git connection
echo -e "\nTesting Git connection..."
git ls-remote https://github.com/hafnium49/GR00T-Dreams.git HEAD 2>&1 | head -1

echo -e "\nTo make DNS changes permanent, add to /etc/wsl.conf:"
echo "[network]"
echo "generateResolvConf = false"