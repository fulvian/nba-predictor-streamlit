#!/usr/bin/env python3
"""
Targeted fix for the _calculate_mapie_confidence_interval function indentation issue
"""

def fix_mapie_function():
    file_path = "src/nba_predictor/core/unified_hybrid_pipeline.py"

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # We need to fix lines 1830-1875 specifically in the _calculate_mapie_confidence_interval function
    # The issue is that these lines should be indented at 12 spaces (inside try block)

    fixed_lines = []
    for i, line in enumerate(lines):
        line_num = i + 1

        if 1830 <= line_num <= 1875:
            # This is the problematic section in _calculate_mapie_confidence_interval
            # All code inside the try block should be indented at 12 spaces
            if line.strip() == "":
                fixed_lines.append(line)  # Keep empty lines as-is
            elif line.strip().startswith('except ') or line.strip().startswith('except:'):
                # Exception handlers should be at 8 spaces (aligned with try)
                fixed_lines.append('        ' + line.strip() + '\n')
            else:
                # Everything else should be at 12 spaces (inside try)
                if line.startswith('          '):  # Currently 10 spaces
                    fixed_lines.append('            ' + line[10:])  # Make it 12 spaces
                elif line.startswith('        '):  # Currently 8 spaces
                    fixed_lines.append('            ' + line[8:])  # Make it 12 spaces
                elif line.startswith('      '):  # Currently 6 spaces
                    fixed_lines.append('            ' + line[6:])  # Make it 12 spaces
                elif line.startswith('    '):  # Currently 4 spaces
                    fixed_lines.append('            ' + line[4:])  # Make it 12 spaces
                else:
                    # Line has no leading spaces, add 12 spaces
                    fixed_lines.append('            ' + line)
        else:
            fixed_lines.append(line)

    # Write back to file
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

    print(f"Fixed indentation in _calculate_mapie_confidence_interval function")

if __name__ == "__main__":
    fix_mapie_function()