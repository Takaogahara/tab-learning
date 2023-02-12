import ray
import argparse
from ray import tune, air
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

from utils import ParametersHandler
from run import train, test


def get_parse():
    parser = argparse.ArgumentParser(description="Tabular Learning")
    parser.add_argument("run", default="train",
                        help="Run type. (train / test)")
    parser.add_argument("--cfg", type=str, required=True,
                        help="Configuration file path")
    parser.add_argument("--train-test", required=False,
                        default=False, help="Finish quickly for testing")

    args, _ = parser.parse_known_args()

    ray.init()

    return args


def trial_name(trial):
    pre = (trial.local_dir).split('/')[-1]
    pos = (trial.trial_id).split('_')[-1]
    return f"{pre}_{pos}"


def create_tuner(parameters, args):
    num_samples = parameters["RUN_RAY_SAMPLES"]
    max_epochs = parameters["DATA_EPOCHS"]
    n_cpus = parameters["RUN_RAY_CPU"]
    n_gpus = parameters["RUN_RAY_GPU"]

    if args.train_test:
        num_samples = 1
        max_epochs = 2
        parameters["DATA_EPOCHS"] = max_epochs
    search_space = ParametersHandler.get_ray_choices(parameters)

    scheduler = AsyncHyperBandScheduler(max_t=max_epochs,
                                        grace_period=1,
                                        reduction_factor=2)

    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=4)

    tuner = tune.Tuner(
        tune.with_resources(train,
                            resources={"cpu": n_cpus, "gpu": n_gpus}),
        run_config=air.RunConfig(name=parameters["RUN_NAME"],
                                 local_dir="./ray_results",
                                 verbose=3),
        tune_config=tune.TuneConfig(metric="loss",
                                    mode="min",
                                    search_alg=algo,
                                    scheduler=scheduler,
                                    num_samples=num_samples,
                                    trial_name_creator=trial_name,
                                    trial_dirname_creator=trial_name),
        param_space=search_space)

    return tuner


if __name__ == "__main__":
    # Get arg parse
    args = get_parse()

    # Load parameters from config file
    parameters = ParametersHandler.load_configs(args.cfg)

    if args.run.lower() == "train":
        # Create Ray Tune instance
        tuner = create_tuner(parameters, args)

        # Run and get results
        results = tuner.fit()
        dataframe = results.get_dataframe()

        # Save
        best_result = results.get_best_result("loss", "min")
        export_path = best_result.log_dir.parent
        csv_path = str(export_path / "results.csv")
        dataframe.to_csv(csv_path, sep=",", index=False)

    elif args.run.lower() == "test":
        _ = test(parameters)
        print("Done")
