#!/usr/bin/env python3
"""
Legacy CLI for document ingestion.
This now delegates to the enhanced ingestion CLI.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Legacy CLI that delegates to enhanced ingestion CLI"""
    print("⚠️  This CLI is deprecated. Use 'npm run ingest' instead.")
    print("   Enhanced features: .txt, .md, .html, .pdf support, block-based processing")
    print()

    # Check if we have arguments to pass through
    if len(sys.argv) > 1:
        print("🔄 Delegating to enhanced ingestion CLI...")
        print()

        # Construct the command for the enhanced CLI
        enhanced_args = ["python", "src/ingest_cli.py"] + sys.argv[1:]

        try:
            # Run the enhanced CLI
            result = subprocess.run(enhanced_args, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"❌ Enhanced CLI failed: {e}")
            return e.returncode
        except FileNotFoundError:
            print("❌ Enhanced ingestion CLI not found: src/ingest_cli.py")
            return 1
    else:
        print("Usage: python cli.py --path <file_or_directory>")
        print("   or: npm run ingest -- --path <file_or_directory>")
        return 1

if __name__ == "__main__":
    sys.exit(main())
