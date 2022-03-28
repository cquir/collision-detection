import os
import time
import click
import numpy as np
import pandas as pd
from collide import collision_detection
import yaml
from tqdm import trange
import uuid

def load_conf(fname : str) -> dict:
    with open(f"../../conf/datagen/{fname}") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def gen_uniform_quaternion() -> np.ndarray:
    """
    convert 3 random uniform numbers between 0 and 1 (u,v,w) 
    to a random uniform quaternion
    """

    a = lambda u,v: np.sqrt(1-u)*np.sin(2*np.pi*v)
    b = lambda u,v: np.sqrt(1-u)*np.cos(2*np.pi*v)
    c = lambda u,w: np.sqrt(u)*np.sin(2*np.pi*w)
    d = lambda u,w: np.sqrt(u)*np.cos(2*np.pi*w)

    [u,v,w] = np.random.uniform(low=0,high=1,size=3)
    return np.array([a(u,v),b(u,v),c(u,w),d(u,w)])

def gen_point() -> np.ndarray:
    """
    Generates a point with shape (14,) in the order:
        | p0 | p1 | q0 | q1 |
    """

    pos0 = np.random.uniform(0,np.sqrt(3),3)
    pos1 = np.random.uniform(0,np.sqrt(3),3)
    q0 = gen_uniform_quaternion()
    q1 = gen_uniform_quaternion()

    return np.concatenate((pos0,q0,pos1,q1), axis=0)


@click.command()
@click.option('--num-bundles', default=1, help='Number of bundles to generate')
@click.option('--num-points', default=1_000, help='Number of points per bundle')
@click.option('--run-id', default="test", help='Where to save the data')
@click.option('--conf-file', default="default.yaml", help='Datagen config')
def main(num_bundles : int, num_points : int, run_id : str, conf_file : str):
    

    conf = load_conf(conf_file)
    np.random.seed(int(conf["seed"]))

    if not os.path.isdir("../../data/datasets"):
        os.mkdir("../../data/datasets")
    
    if not os.path.isdir(f"../../data/datasets/{run_id}"):
        os.mkdir(f"../../data/datasets/{run_id}")

    for _ in range(num_bundles):
        start = time.time()

        data = []
        for _ in trange(num_points):
            # random position + quaternion vectors for each cube
            
            x = gen_point()
            _, res = collision_detection(x)
            
            data.append(np.concatenate( (x, res), axis=0))

        df = pd.DataFrame(data, columns=conf["columns"])
        Y = df["collides"]
        df = df[conf["columns"][:-1]]
        normalized_df = (df-df.min())/(df.max()-df.min())
        print(normalized_df)
        print(f"Generated bundle of {num_points} in {time.time() - start} seconds.")

        # this should likely be separated into a separate preprocessing file / op
        # which is linked closer to training than datagen

        normalized_df["collides"] = Y
        normalized_df.to_parquet(f"../../data/datasets/{run_id}/{len(df)}-{uuid.uuid4()}.parquet")

if __name__ == "__main__":
    main()
