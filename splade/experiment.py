# Implementation of the experiments in the paper SPLADE v2: Sparse Lexical and
# Expansion Model for Information Retrieval, (Thibault Formal, Carlos Lassance,
# Benjamin Piwowarski, Stéphane Clinchant), 2021
# https://arxiv.org/abs/2109.10086

import logging
from functools import partial

import xpmir.interfaces.anserini as anserini
from configuration import SPLADE
from datamaestro_text.data.ir import DocumentStore, FileAccess
from experimaestro.launcherfinder import find_launcher
from xpm_torch.batchers import PowerAdaptativeBatcher
from xpm_torch.learner import Learner
from xpm_torch.trainers.batchwise import BatchwiseTrainer
from xpm_torch.losses.batchwise import SoftmaxCrossEntropy
from xpmir.datasets.adapters import RetrieverBasedCollection
from xpmir.experiments.ir import IRExperimentHelper, ir_experiment
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler
from xpmir.letor.validation import ValidationListener
from xpmir.neural.splade import spladeV2_max
from xpmir.text.huggingface.base import LoadFromHFCheckpoint
from xpmir.papers.helpers.samplers import (
    msmarco_hofstaetter_ensemble_hard_negatives,
    msmarco_v1_docpairs_sampler,
    msmarco_v1_tests,
    msmarco_v1_validation_dataset,
    prepare_collection,
)
from xpmir.papers.results import PaperResults
from xpmir.rankers.full import FullRetriever, FullRetrieverRescorer
from xpmir.rankers.standard import BM25

logging.basicConfig(level=logging.INFO)

# Run by:
# $ experimaestro run-experiment ...


@ir_experiment()
def run(xp: IRExperimentHelper, cfg: SPLADE) -> PaperResults:
    """SPLADE model"""

    gpu_launcher_learner = find_launcher(cfg.splade.requirements)
    gpu_launcher_retrieval = find_launcher(cfg.retrieval.requirements)
    cpu_launcher_index = find_launcher(cfg.indexation.requirements)
    gpu_launcher_index = find_launcher(cfg.indexation.training_requirements)

    random = cfg.random

    # Use MS-Marco (in-memory amounts to 3.6GiB)
    documents: DocumentStore = prepare_collection("irds.msmarco-passage.documents")
    documents.file_access = FileAccess.MMAP
    ds_val_all = msmarco_v1_validation_dataset(cfg.validation)

    tests = msmarco_v1_tests(cfg.dev_test_size)

    # -----The baseline------
    base_model = BM25.C()
    index_builder = anserini.index_builder(launcher=cfg.indexation.launcher)

    retrievers = partial(
        anserini.retriever,
        index_builder,
        model=base_model,
    )  #: Anserini based retrievers

    tests.evaluate_retriever(retrievers, cpu_launcher_index)

    # Building the validation set of the splade
    ds_val = RetrieverBasedCollection.C(
        dataset=ds_val_all,
        retrievers=[retrievers(ds_val_all.documents, k=cfg.retrieval.retTopK)],
    ).submit(launcher=cpu_launcher_index)
    ds_val.documents.in_memory = True

    # -----Learning to rank component preparation part-----
    # Define the model and the flop loss for regularization

    if cfg.splade.model == "splade_max":
        spladev2, flops = spladeV2_max(
            cfg.splade.lambda_q,
            cfg.splade.lambda_d,
            cfg.splade.lambda_warmup_steps,
            cfg.base_hf_id,
        )
    else:
        raise NotImplementedError(f"Cannot handle {cfg.splade.model}")

    # Create the init task to load pretrained HF weights
    hf_encoder = spladev2.encoder.encoder  # HFMaskedLanguageModel (shared)
    load_hf = LoadFromHFCheckpoint.C(model=hf_encoder)

    # Sampler
    if cfg.splade.dataset == "":
        splade_sampler = PairwiseInBatchNegativesSampler.C(
            sampler=msmarco_v1_docpairs_sampler(
                sample_rate=cfg.splade.sample_rate, sample_max=cfg.splade.sample_max
            )
        )

        batchwise_trainer_flops = BatchwiseTrainer.C(
            batch_size=cfg.splade.optimization.batch_size,
            sampler=splade_sampler,
            lossfn=SoftmaxCrossEntropy.C(),
            hooks=[flops],
        )
    elif cfg.splade.dataset == "hofstaetter_kd_hard_negatives":
        batchwise_trainer_flops = DistillationPairwiseTrainer.C(
            batch_size=cfg.splade.optimization.batch_size,
            sampler=msmarco_hofstaetter_ensemble_hard_negatives(),
            lossfn=MSEDifferenceLoss.C(),
            hooks=[flops],
        )

    # establish the validation listener
    validation = ValidationListener.C(
        id="bestval",
        dataset=ds_val,
        retriever=FullRetrieverRescorer.C(
            documents=ds_val.documents,
            scorer=spladev2,
            batchsize=cfg.retrieval.batch_size_full_retriever,
            batcher=PowerAdaptativeBatcher.C(),
        ),
        early_stop=cfg.splade.early_stop,
        validation_interval=cfg.splade.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG@10": False},
    )

    # the learner: Put the components together
    learner = Learner.C(
        random=random,
        trainer=batchwise_trainer_flops,
        model=spladev2,
        optimizers=cfg.splade.optimization.optimizer,
        steps_per_epoch=cfg.splade.optimization.steps_per_epoch,
        max_epochs=cfg.splade.optimization.max_epochs,
        listeners=[validation],
        precision="16-mixed",
    ).tag("model", "splade-v2")

    # submit the learner and build the symbolique link
    outputs = learner.submit(launcher=gpu_launcher_learner, init_tasks=[load_hf])
    xp.tensorboard_service.add(learner, learner.logpath)

    # get the trained model
    load_model = outputs.listeners["bestval"]["RR@10"]

    # build a retriever for the documents
    sparse_index = SparseRetrieverIndexBuilder.C(
        batch_size=512,
        encoder=spladev2.encoder,
        documents=documents,
        ordered_index=False,
        max_docs=cfg.indexation.max_docs,
    ).submit(launcher=gpu_launcher_index, init_tasks=[load_model])

    # Build the sparse retriever based on the index
    splade_retriever = SparseRetriever.C(
        index=sparse_index,
        topk=cfg.retrieval.topK,
        batchsize=1,
        encoder=spladev2.query_encoder,
    )

    # evaluate the best model
    tests.evaluate_retriever(
        splade_retriever,
        gpu_launcher_retrieval,
        model_id=f"{cfg.splade.model}-{cfg.splade.dataset}-RR@10",
        init_tasks=[load_model],
    )

    return PaperResults(
        models={f"{cfg.splade.model}-{cfg.splade.dataset}-RR@10": load_model},
        evaluations=tests,
        tb_logs={f"{cfg.splade.model}-{cfg.splade.dataset}-RR@10": learner.logpath},
    )
