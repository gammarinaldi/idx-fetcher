import os
import subprocess
import time
import socket
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def is_port_open(port, host='127.0.0.1'):
    """Check if a port is open on a host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError):
            return False

def start_ssh_tunnel():
    """Start the SSH tunnel if configured and not already running."""
    use_ssh_tunnel = os.getenv('USE_SSH_TUNNEL', 'FALSE').upper() == 'TRUE'
    if not use_ssh_tunnel:
        return True

    local_port = int(os.getenv('MONGODB_LOCAL_PORT', 27017))
    remote_port = int(os.getenv('MONGODB_REMOTE_PORT', 27017))
    ssh_host = os.getenv('SSH_HOST')
    ssh_user = os.getenv('SSH_USER', 'ubuntu')
    ssh_key_path = os.getenv('SSH_KEY_PATH')
    
    if not ssh_host or not ssh_key_path:
        logger.error("SSH_HOST or SSH_KEY_PATH not configured for SSH tunnel")
        return False

    if is_port_open(local_port):
        logger.info(f"Port {local_port} is already open. Assuming tunnel is running.")
        return True

    logger.info(f"Starting SSH tunnel: {local_port}:localhost:{remote_port} on {ssh_user}@{ssh_host}")
    
    # Expand ~ in ssh_key_path if needed
    if ssh_key_path.startswith('~'):
        ssh_key_path = os.path.expanduser(ssh_key_path)

    # Command for SSH tunnel
    # -N: Do not execute a remote command
    # -f: Go to background (might not work well with subprocess on Windows, 
    # but we'll try to use start on Windows or just run in background)
    ssh_cmd = [
        'ssh', '-i', ssh_key_path,
        '-L', f'{local_port}:localhost:{remote_port}',
        '-N', '-o', 'ExitOnForwardFailure=yes',
        f'{ssh_user}@{ssh_host}'
    ]
    
    try:
        # On Windows, we might want to use 'start' or just Popen
        if os.name == 'nt':
            # Use subprocess.Popen to run in background without blocking
            subprocess.Popen(ssh_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0)
        else:
            # On Unix-like, we can use the -f flag
            ssh_cmd.append('-f')
            subprocess.run(ssh_cmd, check=True)
        
        # Give it a moment to establish
        time.sleep(2)
        
        if is_port_open(local_port):
            logger.info("SSH tunnel established successfully")
            return True
        else:
            logger.error("Failed to establish SSH tunnel")
            return False
            
    except Exception as e:
        logger.error(f"Error starting SSH tunnel: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_ssh_tunnel()
