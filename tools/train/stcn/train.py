import matplotlib as mpl

mpl.use("Agg")
from theseus.opt import Opts
from source.semantic2D.pipeline import Pipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = Pipeline(opts)
    train_pipeline.fit()
