# name -> (module, class). Imported lazily so running one model never requires
# the dependencies of the others.
MODEL_REGISTRY = {
    "StreamingVLM": ("model_wrappers.streamingvlm", "StreamingVLM"),
    "MMDuet2": ("model_wrappers.mmduet2", "MMDuet2"),
    "JoyAI_VL": ("model_wrappers.joyaivl", "JoyAI_VL"),
    "QwenVL": ("model_wrappers.qwen_vl", "QwenVL"),
    # register your model here, e.g.:
    # "MyModel": ("model_wrappers.mymodel", "MyModel"),
}

MODELS = sorted(MODEL_REGISTRY)


def load_model(name: str, model_args):
    import importlib

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: {name}. Registered models: {', '.join(MODELS)}"
        )

    module_path, class_name = MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(model_args)
