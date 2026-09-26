"""
Every scoring threshold, in one place.

Distances are metres and times seconds; positions from the game are converted with ``units_per_metre``. The
defaults are generic and public. Operators are expected to tune their own values in the ``scoring:`` section of
their private config.yaml, so knowing this file does not tell a cheater exactly where the lines are.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Positive = Annotated[float, Field(gt=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- data ---
    units_per_metre: Positive = 100.0  # Unreal units are centimetres
    # Evidence is extracted from consecutive windows of this length, each read with this much leading context
    # (for look-backs and the null) that is not itself counted. Scores use the evidence summed over the horizon.
    window_minutes: Positive = 120.0
    context_minutes: Annotated[float, Field(ge=0)] = 30.0
    horizon_days: Positive = 7.0
    resample_s: Positive = 5.0  # regular grid every trajectory is interpolated onto
    max_gap_s: Positive = 30.0  # longer gaps between samples mean "not present", not "interpolate"

    # --- shared ---
    # How far a player can plausibly see, hear or smell another. Beyond this, heading for someone needs
    # information the game does not give you.
    awareness_m: Positive = 300.0
    # Players who spend this long within this radius of each other in the window are treated as a group (friends
    # on voice chat); heading for a group mate is never evidence.
    associate_radius_m: Positive = 100.0
    associate_min_s: Positive = 600.0
    # The null: other players' trajectories shifted in time by these amounts. It keeps where people tend to go
    # (waterholes, trails) but breaks where they are *now*, which is what ESP reveals.
    null_shifts_s: tuple[Positive, ...] = (600.0, 1200.0, 1800.0)

    # --- beeline: heading straight for a player who was beyond awareness range, and arriving ---
    # Heading = displacement over the next this-many seconds. Short, because a pursuer's heading lags a target
    # that moves sideways.
    heading_window_s: Positive = 15.0
    min_speed_mps: Positive = 1.5  # slower than this counts as not moving
    beeline_cos: Fraction = 0.95  # heading within ~18 degrees of the bearing to the target
    # Aligned for at least this long *while the target is still out of range*. Roaming that happens to point at
    # someone and then turns into a chase once they are in sight lines up only briefly beforehand.
    beeline_min_duration_s: Positive = 60.0
    beeline_bridge_s: Positive = 15.0  # brief heading wobbles up to this long do not end an episode
    beeline_max_start_m: Positive = 3000.0
    # The episode counts once the player has stayed lined up on the target all the way until the target comes
    # into range (within this distance). Evidence stops there: whatever happens after sighting, a charge or a
    # fight, is legitimate, and the time-shifted null could not reproduce a reaction to someone real anyway.
    beeline_arrive_m: Positive = 300.0
    beeline_arrive_grace_s: Positive = 15.0  # the aligned run may end this long before arrival
    # ...and only if the player *turned* onto the target: if this long before the episode they were already
    # heading for the place where they meet, they were going there anyway and the target got in their way (an
    # ambusher standing on their route, someone resting at the waterhole they were walking to).
    beeline_turn_lookback_s: Positive = 60.0
    # ...and only if the target did not come to the player: if the target moved more than this fraction of the
    # start distance towards where the player started, the target's movement explains the meeting. This covers
    # people meeting head-on on a trail, and victims of cheaters (who are approached, or walk into an ambush).
    beeline_max_target_approach: Fraction = 0.3
    # The target must not have been within awareness this recently: following someone you saw a few minutes ago
    # (tracks, scent) is legitimate.
    near_lookback_s: Positive = 300.0
    beeline_min_moving_s: Positive = 600.0  # evidence gate: at least this much moving time in the window
    # Count z-score (observed vs null episodes): the sub-score rises linearly from 0 at the floor to 1 at full.
    # A z of 2 happens by chance for about one player in fifty, so it must not count for anything on its own.
    beeline_z_floor: float = 2.0
    beeline_z_full: Positive = 6.0

    # --- time to contact: how quickly a freshly spawned player reaches someone, versus the org's baseline ---
    contact_m: Positive = 50.0
    respawn_jump_m: Positive = 300.0  # a jump this far between grid steps is a respawn, not movement
    ttc_min_observed_s: Positive = 60.0  # shorter spawn episodes say nothing either way
    ttc_min_episodes: Annotated[int, Field(ge=1)] = 2
    ttc_min_baseline: Annotated[int, Field(ge=1)] = 10  # below this, compare against all classes
    ttc_z_floor: float = 1.0  # rank z-score: sub-score 0 at the floor, 1 at full
    ttc_z_full: Positive = 3.5

    # --- ambush: waiting where a distant player later arrives ---
    stationary_speed_mps: Positive = 0.5
    ambush_min_wait_s: Positive = 60.0
    ambush_radius_m: Positive = 50.0
    ambush_grace_s: Positive = 30.0  # arrivals just after the wait ends still count
    ambush_min_waits: Annotated[int, Field(ge=1)] = 3
    ambush_z_floor: float = 2.0  # count z-score (observed vs null hits): sub-score 0 at the floor, 1 at full
    ambush_z_full: Positive = 6.0

    # --- combining ---
    # Noisy-OR: score = 1 - prod(1 - weight * sub_score). A strong signal in any one behaviour is enough, and
    # time-to-contact alone (weight below the threshold) never is: it only corroborates.
    weight_beeline: Fraction = 1.0
    weight_ttc: Fraction = 0.35
    weight_ambush: Fraction = 0.9
    flag_threshold: Fraction = 0.6
    false_positive_suppress_days: Positive = 30.0
