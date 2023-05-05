import argparse 
import json
import ast

from pathlib import Path

import pandas as pd
import wandb

from nle.dataset import dataset
from nle.dataset import db
from nle.dataset import populate_db


def get_dataset_stats(dataset_name, dbfilename=db.DB):
    sql_args = (dataset_name,)
    
    sql = """
    SELECT games.gameid, games.points, games.turns
    FROM games
    INNER JOIN datasets ON games.gameid=datasets.gameid
    WHERE datasets.dataset_name=?"""

    with db.connect(dbfilename) as conn:
        data = list(conn.execute(sql, sql_args))
        stats = pd.DataFrame(data, columns=["gameid", "points", "turns"])
    return stats


def get_episode_stats(entity, project, run_id):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # print summary of the run just to confirm that this is the run we are interested about
    print(json.dumps(ast.literal_eval(run.summary.__repr__()), sort_keys=True, indent=4))

    artifact = api.artifact(f"{entity}/{project}/run-{run_id}-frame:v0")
    artifact_path = artifact.download()

    with open(Path(artifact_path) / "frame.table.json") as file:
        json_dict = json.load(file)

    df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
    frame = df.to_dict(orient="list")

    game_points, game_turns = frame["scores"], frame["times"]
    return game_points, game_turns


def main(dbfilename: Path, dataset_name: str, entity: str, project: str, run_id: str):
    if dbfilename.exists():
        dataset_stats = get_dataset_stats(dataset_name, str(dbfilename))

        game_points, game_turns = get_episode_stats(entity, project, run_id)
        game_stats = list(zip(game_points, game_turns))

        print(f"games in database: {len(dataset_stats)}")
        with db.db(filename=str(dbfilename), rw=True) as conn:
            games_before = db.count_games(dataset_name, conn)
            for index, row in dataset_stats.iterrows():
                # if (row["points"], row["turns"]) not in game_stats:
                if int(row["points"]) not in game_points:
                    db.drop_games(dataset_name, int(row["gameid"]), conn=conn, commit=True)

            games_left = db.count_games(dataset_name, conn)
            print(f"dropped {games_before - games_left}")
            print(f"left {games_left}")
    else:
        print("database doesn't exist")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbfilename", type=Path)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--entity", type=str, default="gmum")
    parser.add_argument("--project", type=str, default="nle")
    parser.add_argument("--run_id", type=str)

    return parser.parse_known_args(args=args)[0]


if __name__ == "__main__":
    args = vars(parse_args())
    main(**args)
