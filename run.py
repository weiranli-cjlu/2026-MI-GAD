from migad.config import parse_args
from migad.train import run_experiment


def main() -> None:
    config = parse_args()
    run_experiment(config)


if __name__ == "__main__":
    main()
