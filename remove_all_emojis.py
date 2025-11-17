"""
FX-Ai Emoji Removal Script
Automatically removes all emoji characters from Python files and replaces with ASCII
"""

import re
from pathlib import Path

# Emoji to ASCII mapping
EMOJI_MAP = {
    '[SEARCH]': '[SEARCH]',
    '[CHART]': '[CHART]',
    '[TIME]': '[TIME]',
    '[TARGET]': '[TARGET]',
    '[UP]': '[UP]',
    '[DOWN]': '[DOWN]',
    '[CYCLE]': '[CYCLE]',
    '[BLOCKED]': '[BLOCKED]',
    '[TRADE]': '[TRADE]',
    '[PASS]': '[PASS]',
    '[FAIL]': '[FAIL]',
    '[CLOCK]': '[CLOCK]',
    '[LIST]': '[LIST]',
    '[TIMER]': '[TIMER]',
    '[MONEY]': '[MONEY]',
    '[HOT]': '[HOT]',
    '[WARN]': '[WARN]',
    '[STOP]': '[STOP]',
    '[OK]': '[OK]',
    '[ERROR]': '[ERROR]',
    '[WARN]': '[WARN]',
    '[INFO]': '[INFO]',
    '[SUCCESS]': '[SUCCESS]',
    '[PAUSE]': '[PAUSE]',
    '[START]': '[START]',
    '[STOP]': '[STOP]',
    '[STATUS]': '[STATUS]',
    '[HEARTBEAT]': '[HEARTBEAT]',
    '[LONG]': '[LONG]',
    '[SHORT]': '[SHORT]',
    '[OK]': '[OK]',
    '[X]': '[X]',
    '*': '*',
}

def remove_emojis_from_file(file_path):
    """Remove all emojis from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace emojis with ASCII equivalents
        for emoji, replacement in EMOJI_MAP.items():
            content = content.replace(emoji, replacement)

        # Also remove any remaining emoji characters (catch-all)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002700-\U000027BF"  # dingbats
            "\U0001f926-\U0001f937"  # gestures
            "\U00010000-\U0010ffff"  # other unicode
            "\u2640-\u2642"  # gender symbols
            "\u2600-\u2B55"  # misc symbols
            "\u200d"  # zero width joiner
            "\u23cf"  # eject symbol
            "\u23e9"  # fast forward
            "\u231a"  # watch
            "\ufe0f"  # variation selector
            "\u3030"  # wavy dash
            "]+",
            flags=re.UNICODE
        )
        content = emoji_pattern.sub('[EMOJI]', content)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return False

def main():
    """Main function to remove emojis from all Python files"""
    print("FX-Ai Emoji Removal Tool")
    print("=" * 50)

    # Get confirmation
    response = input("This will remove all emojis from Python files. Continue? (yes/no): ").lower().strip()

    if response != 'yes':
        print("Operation cancelled.")
        return

    # Find all Python files
    python_files = []
    for ext in ['*.py']:
        python_files.extend(Path('.').rglob(ext))

    print(f"Found {len(python_files)} Python files")

    updated_count = 0
    for file_path in python_files:
        if remove_emojis_from_file(file_path):
            updated_count += 1

    print(f"\nComplete! Updated {updated_count} files")
    print("Emoji removal finished.")

if __name__ == "__main__":
    main()