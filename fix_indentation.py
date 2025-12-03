#!/usr/bin/env python3
"""
Script to fix indentation issues in unified_hybrid_pipeline.py
"""

def fix_indentation():
    file_path = "src/nba_predictor/core/unified_hybrid_pipeline.py"

    with open(file_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for i, line in enumerate(lines):
        line_num = i + 1

        # Fix the specific issue around line 1830
        if line_num >= 1830 and line_num <= 1875:
            # This is the problematic section in the _calculate_mapie_confidence_interval function
            if line.startswith('          if') or line.startswith('          #') or line.startswith('          base_model') or line.startswith('          mapie_reg') or line.startswith('          return'):
                # Change from 10 spaces to 12 spaces
                if line.startswith('          '):
                    fixed_lines.append('    ' + line[10:])  # Add 2 more spaces
                else:
                    fixed_lines.append(line)
            elif line.startswith('              ') or line.startswith('                ') or line.startswith('                    '):
                # These are already correctly indented (14, 16, 20 spaces)
                fixed_lines.append(line)
            elif line.strip().startswith('except ') or line.strip() == 'except Exception as e:':
                # Exception handlers should be at 8 spaces (aligned with try)
                fixed_lines.append('        ' + line.strip() + '\n')
            elif line.strip().startswith('return None') and line_num == 1875:
                # The final return should be at 8 spaces (function level)
                fixed_lines.append('        return None\n')
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    # Write back to file
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_indentation()