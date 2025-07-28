# Implementation of the experiments in the paper SPLADE v2: Sparse Lexical and
# Expansion Model for Information Retrieval, (Thibault Formal, Carlos Lassance,
# Benjamin Piwowarski, Stéphane Clinchant), 2021
# https://arxiv.org/abs/2109.10086

import logging
from functools import partial

import xpmir.interfaces.anserini as anserini
from configuration import SPLADE
from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher
from xpmir.datasets.adapters import RetrieverBasedCollection
from xpmir.distributed import DistributedHook
from xpmir.experiments.ir import IRExperimentHelper, ir_experiment
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.learner import Learner
from xpmir.text.adapters import TopicTextConverter
from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)
from xpmir.letor.learner import ValidationListener
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler
from xpmir.letor.trainers.batchwise import BatchwiseTrainer, SoftmaxCrossEntropy
from xpmir.neural.dual import DotDense, ScheduledFlopsRegularizer
from xpmir.neural.splade import MaxAggregation, SpladeTextEncoderV2
from xpmir.papers.helpers.samplers import (
    msmarco_hofstaetter_ensemble_hard_negatives,
    msmarco_v1_docpairs_sampler,
    msmarco_v1_tests,
    msmarco_v1_validation_dataset,
    prepare_collection,
)
from xpmir.papers.results import PaperResults
from xpmir.rankers.full import FullRetriever
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import (
    HFStringTokenizer,
    HFTokenizer,
    HFTokenizerAdapter,
    HFMaskedLanguageModel,
)

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

    device = cfg.device
    random = cfg.random

    documents = prepare_collection("irds.msmarco-passage.documents")
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
    # We cannot use the full document dataset to build the validation set.

    # This one could be generic for both sparse and dense methods

    ds_val = RetrieverBasedCollection.C(
        dataset=ds_val_all,
        retrievers=[retrievers(ds_val_all.documents, k=cfg.retrieval.retTopK)],
    ).submit(launcher=cpu_launcher_index)
    ds_val.documents.in_memory = True

    # Base retrievers for validation
    # It retrieve all the document of the collection with score 0
    base_retriever_full = FullRetriever.C(documents=ds_val.documents)

    # -----Learning to rank component preparation part-----
    # Define the model and the flop loss for regularization
    # Model of class: DotDense()
    # The parameters are the regularization coeff for the query and document

    flops = ScheduledFlopsRegularizer.C(
        lambda_q=cfg.splade.lambda_q,
        lambda_d=cfg.splade.lambda_d,
        lambda_warmup_steps=cfg.splade.lambda_warmup_steps,
    )
    tokenizer = HFTokenizer.C(model_id=cfg.base_hf_id)

    if cfg.splade.model == "splade_max":
        splade_encoder = SpladeTextEncoderV2.C(
            tokenizer=HFTokenizerAdapter.C(
                tokenizer=tokenizer, converter=TopicTextConverter.C()
            ),
            encoder=HFMaskedLanguageModel.from_pretrained_id(cfg.base_hf_id),
            aggregation=MaxAggregation.C(),
            maxlen=256,
        )
        spladev2 = DotDense.C(encoder=splade_encoder)
    else:
        raise NotImplementedError(f"Cannot handle {cfg.splade.model}")

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

    # hooks for the learner
    if cfg.splade.model == "splade_doc" or spladev2.query_encoder is None:
        hooks = [
            setmeta(
                DistributedHook.C(models=[spladev2.encoder]),
                True,
            )
        ]
    else:
        hooks = [
            setmeta(
                DistributedHook.C(models=[spladev2.encoder, spladev2.query_encoder]),
                True,
            )
        ]

    # establish the validation listener
    validation = ValidationListener.C(
        id="bestval",
        dataset=ds_val,
        # a retriever which use the splade model to score all the
        # documents and then do the retrieve
        retriever=spladev2.getRetriever(
            base_retriever_full,
            cfg.retrieval.batch_size_full_retriever,
            PowerAdaptativeBatcher.C(),
            device=device,
        ),
        early_stop=cfg.splade.early_stop,
        validation_interval=cfg.splade.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG@10": False},
    )

    # the learner: Put the components together
    learner = Learner.C(
        # Misc settings
        random=random,
        device=device,
        # How to train the model
        trainer=batchwise_trainer_flops,
        # the model to be trained
        model=spladev2,
        # Optimization settings
        optimizers=cfg.splade.optimization.optimizer,
        steps_per_epoch=cfg.splade.optimization.steps_per_epoch,
        use_fp16=True,
        max_epochs=cfg.splade.optimization.max_epochs,
        # the listener for the validation
        listeners=[validation],
        # the hooks
        hooks=hooks,
    ).tag("model", "splade-v2")

    # submit the learner and build the symbolique link
    outputs = learner.submit(launcher=gpu_launcher_learner)
    xp.tensorboard_service.add(learner, learner.logpath)

    # get the trained model
    load_model = (
        outputs.learned_model
        if cfg.splade.model == "splade_doc"
        else outputs.listeners["bestval"]["RR@10"]
    )

    # build a retriever for the documents
    sparse_index = SparseRetrieverIndexBuilder.C(
        batch_size=512,
        batcher=PowerAdaptativeBatcher.C(),
        encoder=spladev2.encoder,
        device=device,
        documents=documents,
        ordered_index=False,
        max_docs=cfg.indexation.max_docs,
    ).submit(launcher=gpu_launcher_index, init_tasks=[load_model])

    # Build the sparse retriever based on the index
    splade_retriever = SparseRetriever.C(
        index=sparse_index,
        topk=cfg.retrieval.topK,
        batchsize=1,
        encoder=spladev2._query_encoder,
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
