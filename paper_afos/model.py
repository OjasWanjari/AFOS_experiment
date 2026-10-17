"""Paper Table 2 CNN and the complete Section 3.1.1 function vocabulary."""
import math
import numpy as np
import tensorflow as tf
from .search import tree


def evaluate_tree(node, x):
    if node[0] == "u":
        operations = {
            "x": lambda: x, "neg": lambda: -x, "exp": lambda: tf.exp(x),
            "abs": lambda: tf.abs(x), "exp_neg": lambda: tf.exp(-x),
            "min_zero": lambda: tf.minimum(x, 0), "relu": lambda: tf.maximum(x, 0),
            "sin": lambda: tf.sin(x), "cos": lambda: tf.cos(x), "sinh": lambda: tf.sinh(x),
            "tanh": lambda: tf.tanh(x), "asin": lambda: tf.asin(x), "atan": lambda: tf.atan(x),
            "asinh": lambda: tf.asinh(x), "atanh": lambda: tf.atanh(x), "erf": lambda: tf.math.erf(x),
            "sigmoid": lambda: tf.sigmoid(x), "x_erf": lambda: x * tf.math.erf(x),
            "softplus": lambda: tf.nn.softplus(x),
        }
        return operations[node[1]]()
    a, b = evaluate_tree(node[2], x), evaluate_tree(node[3], x)
    operations = {
        "add": lambda: a + b, "mul": lambda: a * b,
        "max": lambda: tf.maximum(a, b), "min": lambda: tf.minimum(a, b),
        "max_relu": lambda: tf.maximum(tf.maximum(a, 0), b),
        "max_minzero": lambda: tf.maximum(tf.minimum(a, 0), b),
        "min_relu": lambda: tf.minimum(tf.maximum(a, 0), b),
        "min_minzero": lambda: tf.minimum(tf.minimum(a, 0), b),
        "mul_exp": lambda: a * tf.exp(b), "mul_sigmoid": lambda: a * tf.sigmoid(b),
        "mul_erf": lambda: a * tf.math.erf(b),
    }
    return operations[node[1]]()


@tf.keras.utils.register_keras_serializable(package="paper_afos")
class PaperActivation(tf.keras.layers.Layer):
    def __init__(self, chromosome, **kwargs):
        super().__init__(**kwargs)
        self.chromosome = list(chromosome)
        self.activation_tree = tree(self.chromosome)

    def call(self, inputs):
        return evaluate_tree(self.activation_tree, inputs)

    def get_config(self):
        return dict(super().get_config(), chromosome=self.chromosome)


@tf.keras.utils.register_keras_serializable(package="paper_afos")
def mish(x):
    return x * tf.tanh(tf.nn.softplus(x))


@tf.keras.utils.register_keras_serializable(package="paper_afos")
def serf(x):
    return x * tf.math.erf(tf.nn.softplus(x))


def build_model(config, chromosome=None, baseline=None, jit_compile=None):
    if (chromosome is None) == (baseline is None):
        raise ValueError("Supply exactly one chromosome or baseline")
    training, cnn = config["training"], config["cnn"]
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(32, 32, 3))])
    baselines = {"relu": tf.nn.relu, "swish": tf.nn.swish, "gelu": tf.nn.gelu,
                 "mish": mish, "serf": serf, "elu": tf.nn.elu}
    for i, filters in enumerate(cnn["filters"]):
        model.add(tf.keras.layers.Conv2D(filters, 3, padding=training["padding"],
                                        kernel_initializer=training["kernel_initializer"]))
        # Parameter-free layer: no extra channel-mixing Dense layers.
        model.add(PaperActivation(chromosome) if chromosome is not None
                  else tf.keras.layers.Activation(baselines[baseline]))
        if i % 2 == 1:
            model.add(tf.keras.layers.MaxPooling2D(2))
            model.add(tf.keras.layers.Dropout(cnn["dropout_rate"]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax", kernel_initializer=training["kernel_initializer"]))
    compile_options = {} if jit_compile is None else {"jit_compile": jit_compile}
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=training["learning_rate"],
                                                   momentum=training["momentum"]),
                  loss=training["loss"], metrics=["accuracy"], **compile_options)
    return model


class StopNonFinite(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.failed = False

    def on_train_batch_end(self, batch, logs=None):
        if not math.isfinite(float((logs or {}).get("loss", 0))):
            self.failed = True
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        if any(not math.isfinite(float(v)) for k, v in (logs or {}).items()
               if k in ("loss", "accuracy", "val_loss", "val_accuracy")):
            self.failed = True
            self.model.stop_training = True


def train_candidate(model, train, validation, config, search=True, verbose=1, on_epoch=None):
    """Deliberately accepts no test set. No test metrics exist during search."""
    epochs = config["training"]["epochs"] if search else config["post_search"]["epochs"]
    callback = StopNonFinite()
    class RecordEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if on_epoch is not None:
                on_epoch(dict(epoch=epoch + 1, **{k: float(v) for k, v in (logs or {}).items()}))
    callbacks = [callback, RecordEpoch()]
    history = []
    first = model.fit(*train, validation_data=validation, batch_size=config["training"]["batch_size"],
                      epochs=1 if search else epochs, callbacks=callbacks, verbose=verbose)
    def collect(h, offset=0):
        for j in range(len(h.history["loss"])):
            history.append(dict(epoch=offset + j + 1, **{k: float(v[j]) for k, v in h.history.items()}))
    collect(first)
    status = "completed"
    if callback.failed:
        status = "nonfinite"
    elif search and history[-1]["val_accuracy"] <= config["training"]["first_epoch_accuracy_threshold"]:
        status = "early_stopped"
    elif search and epochs > 1:
        rest = model.fit(*train, validation_data=validation, batch_size=config["training"]["batch_size"],
                         initial_epoch=1, epochs=epochs, callbacks=callbacks, verbose=verbose)
        collect(rest, 1)
        if callback.failed:
            status = "nonfinite"
    last = history[-1]
    metrics = {"train_loss": last["loss"], "train_accuracy": last["accuracy"],
               "validation_loss": last["val_loss"], "validation_accuracy": last["val_accuracy"],
               "epochs_completed": len(history), "status": status}
    return metrics, history


def load_data(config, smoke=False, split_file=None):
    if smoke:
        rng = np.random.default_rng(1234)
        def sample(n):
            return rng.random((n, 32, 32, 3), dtype=np.float32), tf.keras.utils.to_categorical(rng.integers(10, size=n), 10)
        return sample(32), sample(16), sample(16), None
    from sklearn.model_selection import train_test_split
    (x, y), (tx, ty) = tf.keras.datasets.cifar10.load_data()
    if split_file is None:
        indices = np.arange(len(x))
        train_idx, val_idx = train_test_split(indices, test_size=config["dataset"]["validation_size"],
                                            random_state=config["dataset"]["split_seed"])
    else:
        # Retraining uses the exact split recorded by the original search.
        with np.load(split_file, allow_pickle=False) as saved:
            train_idx = saved["train_indices"]
            val_idx = saved["validation_indices"]
        combined = np.concatenate((train_idx, val_idx))
        if (train_idx.shape != (config["dataset"]["train_size"],)
                or val_idx.shape != (config["dataset"]["validation_size"],)
                or not np.issubdtype(combined.dtype, np.integer)
                or not np.array_equal(np.sort(combined), np.arange(len(x)))):
            raise ValueError("Saved split must partition all CIFAR-10 training examples exactly once")
    def prep(images, labels):
        return images.astype("float32") / 255.0, tf.keras.utils.to_categorical(labels, 10)
    return (prep(x[train_idx], y[train_idx]), prep(x[val_idx], y[val_idx]), prep(tx, ty),
            {"train_indices": train_idx, "validation_indices": val_idx})
