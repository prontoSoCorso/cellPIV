import os

with open('SUMMARY.md', 'w') as outf:
    outf.write('# Project Summary\n\n')
    for root, dirs, files in os.walk('.'):
        indent = '  ' * (root.count(os.sep))
        outf.write(f'{indent}- **{os.path.basename(root) or "."}**\n')
        for f in sorted(files):
            if f.endswith('.py'):
                outf.write(f'{indent}  - {f}\n')