# Third party imports
import cltoolbox

# First party imports
from hsp2.hsp2tools.commands import (
    convert_control_files,
    import_uci,
    run_cli,
    update_uci,
)


def main():
    cltoolbox.command("run")(run_cli)
    cltoolbox.command(import_uci)
    cltoolbox.command(update_uci)
    cltoolbox.command(convert_control_files)
    cltoolbox.main()


if __name__ == "__main__":
    main()
