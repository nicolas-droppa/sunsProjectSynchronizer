from dataclasses import dataclass

@dataclass
class Config:
    # data
    csvPath: str = "dataCombined/out_data.csv"
    imgFolder: str = "dataCombined/originals"
    imgSize: tuple = (32, 32)
    batchSize: int = 16
    numEpochs: int = 40
    valRatio: float = 0.1
    testRatio: float = 0.1
    numWorkers: int = 4

    # model / training
    learningRate: float = 1e-3
    useL1: bool = False
    l1Lambda: float = 1e-5
    useL2: bool = True
    l2Lambda: float = 1e-4  # weight_decay

    useBatchnorm: bool = True
    useDropout: bool = True
    dropoutP: float = 0.3

    # misc
    seed: int = 42
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    checkpointDir: str = "checkpoints"
    resultsCsv: str = "experiments_results.csv"