from nle.env import tasks


ENVS = dict(
    staircase=tasks.NetHackStaircase,
    score=tasks.NetHackScore,
    pet=tasks.NetHackStaircasePet,
    oracle=tasks.NetHackOracle,
    gold=tasks.NetHackGold,
    eat=tasks.NetHackEat,
    scout=tasks.NetHackScout,
    challenge=tasks.NetHackChallenge,
)
