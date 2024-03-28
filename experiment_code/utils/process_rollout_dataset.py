import argparse 

from shutil import copyfile
from pathlib import Path
from itertools import groupby


def main(input_dir: Path, output_dir: Path):
    """
    input_dir have files without structure. We want to group them and put into unique folders.

    input_dir/
        nle.1284030.0.ttyrec3.bz2
        nle.1284030.1.ttyrec3.bz2
        ...
        nle.1284030.xlogfile
        .
        .
        .
        nle.1284032.0.ttyrec3.bz2
        nle.1284032.1.ttyrec3.bz2
        nle.1284032.xlogfile

    output_dir/
        nle.1284030/
            nle.1284030.0.ttyrec3.bz2
            nle.1284030.1.ttyrec3.bz2
            ...
            nle.1284030.xlogfile
        .
        .
        .
        nle.1284032/
            nle.1284032.0.ttyrec3.bz2
            nle.1284032.1.ttyrec3.bz2
            ...
            nle.1284032.xlogfile
    """

    data = list(input_dir.iterdir())
    # group with first two parts from name
    keyfunc = lambda path: ".".join(str(path.name).split(".")[:2])
    # take int from name
    sortfunc = lambda path: int(str(path.name).split(".")[2])

    print("started copying")

    data = sorted(data, key=keyfunc)
    for k, g in groupby(data, keyfunc):
        g = list(g)

        group_dir = output_dir / k
        group_dir.mkdir(parents=True, exist_ok=True)

        # for e, path in enumerate(sorted(g, key=sortfunc)):
        for e, path in enumerate(g):
            if path.name == "rollout":
                continue

            new_path = group_dir / path.name
            copyfile(path, new_path)

    print("finished copying")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)

    return parser.parse_known_args(args=args)[0]


if __name__ == "__main__":
    args = vars(parse_args())
    main(**args)