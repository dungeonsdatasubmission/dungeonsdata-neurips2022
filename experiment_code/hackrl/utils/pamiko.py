import paramiko



def get_checkpoint_paths(dir):
    # SSH connection details
    host = 'athena.cyfronet.pl'
    username = 'plgbartekcupial'
    private_key_path = '/home/bartek/.ssh/id_rsa'


    # Command to execute
    # dir = '/net/pr2/projects/plgrid/plgg_pw_crl/mostaszewski/mrunner_scratch/nle/06_05-10_51-gallant_sammet'
    find = f'find {dir} -type f -name "checkpoint_*" -printf "%h\n" | sort -u'


    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

        # Connect to the SSH server
        ssh.connect(host, username=username, pkey=private_key)

        # Open an SFTP session
        sftp = ssh.open_sftp()

        # Execute the command
        stdin, stdout, stderr = ssh.exec_command(find)

        # Read the output
        output = stdout.read().decode()

        paths = output.split('\n')
        # filter empty strings
        paths = list(filter(None, paths))

        # Close the SFTP session
        sftp.close()
        return paths
    finally:
        # Close the SSH connection
        ssh.close()
