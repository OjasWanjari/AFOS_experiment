"""Record runtime details and enforce the recorded comparison environment."""
import importlib.metadata
import os
import platform


def inspect_environment(runtime):
    # Configure TensorFlow before importing models or loading any data.
    if runtime["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
    import tensorflow as tf
    tf.config.optimizer.set_jit(False)
    devices = tf.config.list_physical_devices("GPU")
    if runtime["device"] == "cpu":
        tf.config.set_visible_devices([], "GPU")
    elif not devices:
        raise RuntimeError("No TensorFlow GPU found. Use the original GPU environment.")
    else:
        for gpu in devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    versions = {}
    for package in ("tensorflow", "tensorflow-cpu", "keras", "numpy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=versions, tensorflow=tf.__version__, build=tf.sysconfig.get_build_info(),
                devices=[str(d) for d in devices], runtime=runtime,
                gpu_names=[tf.config.experimental.get_device_details(d).get("device_name", "") for d in devices],
                environment={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TF_XLA_FLAGS", "XLA_FLAGS")})


def environment_mismatches(expected, actual):
    """Compare what the original run recorded, without claiming driver identity."""
    mismatches = []
    for key in ("python", "tensorflow", "packages", "runtime"):
        if expected.get(key) != actual.get(key):
            mismatches.append(f"{key}: expected {expected.get(key)!r}; got {actual.get(key)!r}")
    if expected["platform"].split("-")[0] != actual["platform"].split("-")[0]:
        mismatches.append(f"OS family: expected {expected['platform']}; got {actual['platform']}")
    for key in ("cuda_version", "cudnn_version", "is_cuda_build", "is_rocm_build"):
        if expected.get("build", {}).get(key) != actual.get("build", {}).get(key):
            mismatches.append(f"TensorFlow build.{key} differs")
    if len(expected["devices"]) != len(actual["devices"]):
        mismatches.append("Visible GPU count differs")
    for key in ("TF_XLA_FLAGS", "XLA_FLAGS"):
        if expected.get("environment", {}).get(key) != actual.get("environment", {}).get(key):
            mismatches.append(f"{key} differs")
    if expected.get("gpu_names") and expected["gpu_names"] != actual.get("gpu_names"):
        mismatches.append(f"GPU model: expected {expected['gpu_names']}; got {actual.get('gpu_names')}")
    return mismatches
