import logging

from experimaestro import experiment
from xpmir.learning.optim import (
    TensorboardService,
)
import xpmir.evaluation
from xpmir.papers.cli import paper_command
from xpmir.papers.results import PaperResults
from configuration import MyModel

logging.basicConfig(level=logging.INFO)


def run(
    xp: experiment, cfg: MyModel, tensorboard_service: TensorboardService
) -> PaperResults:
    """My model"""

    # A model that can be loaded, i.e. what is
    # returned by a learner or one of its validation listener.
    my_model = ...

    # The submitted learner
    learner = ...

    # The results of evaluations
    tests: xpmir.evaluation.EvaluationsCollection = ...

    return PaperResults(
        models={"MyModel-RR@10": my_model},
        evaluations=tests,
        tb_logs={"MyModel-RR@10": learner.logpath},
    )


@paper_command(schema=MyModel, folder=__file__, tensorboard_service=True)
def cli(xp: experiment, cfg: MyModel, tensorboard_service: TensorboardService):
    return run(xp, cfg, tensorboard_service)


if __name__ == "__main__":
    cli()
