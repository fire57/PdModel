import argparse

from smoke_api import smoke

from pd_model.config import load_config
from pd_model.monitoring.simulate import simulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    smoke(args.url)
    config = load_config()
    simulate(config, "normal", args.url)
    simulate(config, "shifted", args.url)
