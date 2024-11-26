from typing import List, Tuple, Union
import coiled
import dask.config
from dask.distributed import Client, LocalCluster

from stem_continuation_dataset_generator.constants import DASK_CLUSTER_NAME

NUM_WORKERS = [4, 50]


def get_client(
    run_locally: bool = False,
    n_workers: Union[int, List[int]] = NUM_WORKERS,
    **kwargs,
) -> Union[Client, Tuple[Client, str]]:

    dask.config.set({'distributed.scheduler.allowed-failures': 12})

    if run_locally is True:
        cluster = LocalCluster(n_workers=2, threads_per_worker=1)

    else:
        cluster = coiled.Cluster(
            name=DASK_CLUSTER_NAME,
            n_workers=n_workers,
            package_sync_conda_extras=['portaudio', 'ffmpeg'],
            idle_timeout="5 minutes",
            **kwargs,
        )

    client = cluster.get_client()

    return client