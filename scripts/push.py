#!/usr/bin/env python3
import subprocess, os
os.chdir('/home/z/my-project/xorzen-repo')
subprocess.run(['git', 'add', '-A'])
subprocess.run(['git', 'commit', '-m', 'fix: causal mask shape + SSM causal conv + ConfigFactory enum + dead param removal', '--allow-empty', '-q'])
subprocess.run(['git', 'push', 'origin', 'main'])
print('done')
