import argparse 
import time
from pathlib import Path

from nle.dataset import dataset
from nle.dataset import db
from nle.dataset import populate_db


def main(dbfilename: Path, dataset_path: Path, dataset_name: str):
    if not dbfilename.exists():
        db.create(str(dbfilename))

    with db.db(filename=str(dbfilename), rw=True) as conn:
        conn.execute("DELETE FROM datasets WHERE dataset_name=?", (dataset_name,))
        conn.execute("DELETE FROM roots WHERE dataset_name=?", (dataset_name,))
        conn.execute("UPDATE meta SET mtime = ?", (time.time(),))
        conn.commit()

    populate_db.add_nledata_directory(str(dataset_path), dataset_name, str(dbfilename))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbfilename", type=Path)
    parser.add_argument("--dataset_path", type=Path)
    parser.add_argument("--dataset_name", type=str)

    return parser.parse_known_args(args=args)[0]


if __name__ == "__main__":
    args = vars(parse_args())
    main(**args)