from tools.sim.behaviours import BeelineCheater, Behaviour, Roamer
from tools.sim.world import UNREAL_UNITS_PER_METRE, SimPlayer, SnapshotRow, World


def demo_world(n_honest: int = 15, *, cheater: bool = True, seed: int = 0) -> World:
    """
    The world the fake RCON server runs by default: honest roamers sharing a few waterholes, plus one cheater.

    :param n_honest: number of honest players
    :param cheater: whether to add the scripted beeline cheater
    :param seed: rng seed
    :return: the world
    """
    world = World(seed=seed)
    pois = [world.random_point(margin_m=500.0) for _ in range(4)]
    for _ in range(n_honest):
        world.add_player(Roamer(pois=pois, poi_bias=0.5, speed_mps=float(world.rng.uniform(4.0, 8.0))))
    if cheater:
        world.add_player(BeelineCheater(), archetype="cheater", name="Totally, Legit")
    return world


__all__ = [
    "UNREAL_UNITS_PER_METRE",
    "BeelineCheater",
    "Behaviour",
    "Roamer",
    "SimPlayer",
    "SnapshotRow",
    "World",
    "demo_world",
]
