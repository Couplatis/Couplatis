from loguru import logger

import couplatis


def main():
    config = couplatis.config.Config()
    logger.info(f"Running with config: {config}")

    couplatis.run(config)


if __name__ == "__main__":
    main()
