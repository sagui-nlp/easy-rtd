class ModelPaths:
    generator_config = "deberta-v3-xsmall-changed/generator_config.json"
    generator_weights = "deberta-v3-xsmall-changed/pytorch_model.generator.bin"
    discriminator_config = "deberta-v3-xsmall-changed/config.json"
    discriminator_weights = "deberta-v3-xsmall-changed/pytorch_model.bin"


class TrainArgs(ModelPaths):
    per_device_train_batch_size: int = 1
    temperature: float = 1.0
    rtd_lambda: float = 50.0


targs = TrainArgs()
