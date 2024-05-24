import cltoolbox

from hsp2.hsp2tools.commands import import_uci, run


def main():
    cltoolbox.command(run)
    cltoolbox.command(import_uci)
    cltoolbox.main()


if __name__ == "__main__":
    main()
