import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from data_load import app  # noqa: E402


def main() -> None:
    app()


if __name__ == "__main__":
    main()
