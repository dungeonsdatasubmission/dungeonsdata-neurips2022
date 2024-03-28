from nle.dataset import db


def get_dataset_scores(dataset_name, dbfilename=db.DB):
    sql_args = (dataset_name,)
    
    sql = """
    SELECT games.gameid, games.points
    FROM games
    INNER JOIN datasets ON games.gameid=datasets.gameid
    WHERE datasets.dataset_name=?"""

    with db.connect(dbfilename) as conn:

        scores = dict(list(conn.execute(sql, sql_args)))
    return scores
