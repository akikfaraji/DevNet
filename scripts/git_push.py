#!/usr/bin/env python3
"""Push XORZEN fixes to GitHub."""
import subprocess, os, sys

REPO = '/home/z/my-project/xorzen-repo'

def do_push():
    os.chdir(REPO)
    subprocess.run(['git', 'add', '-A'], check=False)
    subprocess.run(['git', 'commit', '-m',
        'fix: causal mask shape + SSM causal conv + ConfigFactory enum + dead param removal',
        '--allow-empty', '-q'], check=False)
    subprocess.run(['git', 'push', 'origin', 'main'], check=False)

if __name__ == '__main__':
    do_push()
