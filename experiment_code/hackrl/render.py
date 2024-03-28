import os

from nle.dataset import dataset
from nle.dataset import db
from nle.dataset import populate_db

from nle import nethack

dbfilename = "./ttyrecs.db"

if not os.path.isfile(dbfilename):
    alt_path = "/nle/nld-nao"
    aa_path = "/nle/nld-aa-taster/nle_data"
    db.create(dbfilename)
    populate_db.add_nledata_directory(aa_path, "autoascend", dbfilename)
    populate_db.add_altorg_directory(alt_path, "altorg", dbfilename)


kwargs = dict(
    batch_size=1,
    seq_length=1,
    dbfilename=dbfilename,
    loop_forever=True,
    shuffle=True,
)

dataset = dataset.TtyrecDataset("autoascend", **kwargs)
for i, mb in enumerate(dataset):

    tty_chars = mb["tty_chars"][0][0]
    tty_colors = mb["tty_colors"][0][0]
    tty_cursor = mb["tty_cursor"][0][0]
    print(nethack.tty_render(tty_chars, tty_colors, tty_cursor))

    if mb["done"][0][0]:
        print("trajectory end: breakpoint")