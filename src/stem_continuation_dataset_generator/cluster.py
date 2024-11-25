from typing import List, Tuple, Union
import coiled
import dask.config
from dask.distributed import Client, LocalCluster

NUM_WORKERS = [4, 50]
BUCKET = 's3://stem-continuation-dataset'


def get_client(
    run_locally: bool = False,
    n_workers: Union[int, List[int]] = NUM_WORKERS,
    **kwargs,
) -> Union[Client, Tuple[Client, str]]:

    dask.config.set({'distributed.scheduler.allowed-failures': 12})

    if run_locally is True:
        cluster = LocalCluster(n_workers=2, threads_per_worker=1, **kwargs)

    else:
        cluster = coiled.Cluster(
            n_workers=n_workers,
            package_sync_conda_extras=['portaudio', 'ffmpeg'],
            idle_timeout="5 minutes",
            **kwargs,
        )

    client = cluster.get_client()

    return client