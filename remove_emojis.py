#!/usr/bin/env python3
"""Remove emojis from markdown files and replace with ASCII"""

import os

# Emoji to ASCII mapping
EMOJI_MAP = {
    '✅': '[+]',
    '❌': '[-]',
    '📊': '[CHART]',
    '🎯': '[TARGET]',
    '🛑': '[STOP]',
    '🔄': '[CYCLE]',
    '💡': '[IDEA]',
    '❓': '[?]',
    '🚀': '[>>]',
    '📈': '[TREND]',
    '📁': '[FILE]',
    '🎉': '[*]',
    '🔍': '[SEARCH]',
    '📝': '[NOTE]',
    '✓': '[OK]',
}

files_to_process = [
    'IMPLEMENTATION_SUMMARY.md',
    'DECISION_FLOW.md',
    'SL_TP_ENHANCEMENTS.md',
]

for filename in files_to_process:
    if not os.path.exists(filename):
        print(f"Skipping {filename} - not found")
        continue
    
    print(f"Processing {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace all emojis
    for emoji, ascii_replacement in EMOJI_MAP.items():
        content = content.replace(emoji, ascii_replacement)
    
    if content != original_content:
        with open(filename, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        print(f"  [+] Updated {filename}")
    else:
        print(f"  [-] No changes needed for {filename}")

print("\nDone!")
