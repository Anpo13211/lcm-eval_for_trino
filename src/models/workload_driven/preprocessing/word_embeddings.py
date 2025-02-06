import logging
import multiprocessing
import os
import time

from gensim.models import Word2Vec

from cross_db_benchmark.benchmark_tools.utils import load_json


def compute_word_embeddings(source, target):
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)

    os.makedirs(os.path.dirname(target), exist_ok=True)
    sentences = load_json(source)
    print(f"Constructing word embeddings for {len(sentences)} sentences")

    # hyperparameters taken from Sun et al.
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=5,
                         window=5,
                         vector_size=500,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores - 2)

    t = time.perf_counter()
    w2v_model.build_vocab(sentences, progress_per=10000)
    print(f"Time to build vocab: {(time.perf_counter() - t) / 60:.2f} mins")

    t = time.perf_counter()
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    print(f"Time to train the model: {(time.perf_counter() - t) / 60:.2f} mins")

    # w2v_model.wv['Carrier_EV']
    w2v_model.wv.save(target)
    print('model saved')
