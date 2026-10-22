from __future__ import annotations

import train_dtd
from materialized_v7_dataset import create_materialized_v7_datasets


def main() -> None:
    train_dtd.create_enriched_datasets = create_materialized_v7_datasets
    train_dtd.main()


if __name__ == "__main__":
    main()
