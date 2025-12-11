# Brawler Operations Guide

## Deployment
**NEVER** restart the service manually if you have changed the service definition (e.g., config file path).
ALWAYS use the deployment script to ensure the systemd definition is updated.

```bash
# Correct way to deploy changes:
./scripts/deploy_brawler.sh
```

## Configuration Checks
Before checking out, verify:
1.  `brawler.service` points to the correct `.yaml` config.
2.  The `.yaml` config has correct assets and limits.

## Emergency Stop
```bash
sudo systemctl stop brawler
```
Or kill manually if hung:
```bash
ps aux | grep brawler
sudo kill -9 <PID>
```
