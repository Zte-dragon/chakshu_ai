# Gunicorn Status and Logging Guide

1. Check Gunicorn Process Status:
   ```
   ps aux | grep gunicorn
   sudo systemctl status gunicorn  # if using systemd
   ```

2. View Gunicorn Logs:
   ```
   # If running with default logging
   tail -f /var/log/gunicorn/access.log  # for access logs
   tail -f /var/log/gunicorn/error.log   # for error logs
   ```

3. To restart Gunicorn:
   ```
   sudo pkill gunicorn  # Stop all Gunicorn processes
   # Then start again with:
   sudo /home/devuser/miniconda3/envs/chaksuai/bin/gunicorn chakshu_ai.wsgi:application --workers 3 --bind 0.0.0.0:80 --daemon --access-logfile /var/log/gunicorn/access.log --error-logfile /var/log/gunicorn/error.log
   ```

4. Monitor Worker Processes:
   ```
   pstree -ap | grep gunicorn
   ```

5. Check Resource Usage:
   ```
   top -p $(pgrep -d',' -f gunicorn)
   ```

6. to clear the log:
   ```
   sudo truncate -s 0 /var/log/gunicorn/access.log /var/log/gunicorn/error.log
   ```
   
   


Note: To set up proper logging, create the log directory first:
```
sudo mkdir -p /var/log/gunicorn
sudo chown -R devuser:devuser /var/log/gunicorn
```

For better process management, consider creating a systemd service file:
```
[Unit]
Description=Gunicorn daemon for Chakshu AI
After=network.target

[Service]
User=devuser
Group=devuser
WorkingDirectory=/home/devuser/Puneet/chakshu1/chakshu_ai/chakshu_ai
ExecStart=/home/devuser/miniconda3/envs/chaksuai/bin/gunicorn \
    --workers 3 \
    --bind 0.0.0.0:80 \
    --access-logfile /var/log/gunicorn/access.log \
    --error-logfile /var/log/gunicorn/error.log \
    chakshu_ai.wsgi:application

[Install]
WantedBy=multi-user.target
```

Save this as: /etc/systemd/system/gunicorn.service
Then enable and start with:
```
sudo systemctl enable gunicorn
sudo systemctl start gunicorn
