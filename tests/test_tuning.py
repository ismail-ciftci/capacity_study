from capacity_study.tuning import Ranges, select_best, trial_objectives


def test_trial_objective_is_mean_over_seeds_and_infeasible_if_any_seed_is():
    assert trial_objectives([[0.1, 0.3], [0.5, None], [None, None]]) == [0.2, None, None]


def test_a_lucky_unstable_trial_does_not_win():
    seeds = [[0.010, 0.012, 0.011, 0.009],      # stable
             [0.030, -5.0, -2.0, -7.0]]          # one lucky seed, the rest blew up
    ranges = Ranges((1e-4, 1e-1), (1e-5, 1.0))
    points = [{"lr": 1e-3, "wd": 0.0}, {"lr": 5e-2, "wd": 0.0}]
    best = select_best(0, "none", points, trial_objectives(seeds), ranges)
    assert best.best_trial == 0 and best.infeasible_fraction == 0.0
