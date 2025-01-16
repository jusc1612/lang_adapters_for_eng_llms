from adapters import AdapterArguments, AdapterConfig
from adapters.composition import Stack
from typing import Optional

def setup_adapter_training_(
    model,
    adapter_args: AdapterArguments,
    adapter_name: str,
    adapter_config_kwargs: Optional[dict] = None,
    adapter_load_kwargs: Optional[dict] = None,
    use_la_dummy_adapter: Optional[bool] = False,
    fusion: Optional[bool] = False,
    fuse_langs: Optional[list] = None, 
):
    """Setup model for adapter training based on given adapter arguments.

    Args:
        model (_type_): The model instance to be trained.
        adapter_args (AdapterArguments): The adapter arguments used for configuration.
        adapter_name (str): The name of the adapter to be added.

    Returns:
        Tuple[str, str]: A tuple containing the names of the loaded adapters.
    """
    if adapter_config_kwargs is None:
        adapter_config_kwargs = {}
    if adapter_load_kwargs is None:
        adapter_load_kwargs = {}
    # Setup adapters
    if adapter_args.train_adapter and not fusion:
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(adapter_args.lang_adapter_config, **adapter_config_kwargs)
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                with_head=False,
                **adapter_load_kwargs,
            )
        elif use_la_dummy_adapter:
            lang_adapter_config = AdapterConfig.load(adapter_args.adapter_config, **adapter_config_kwargs)
            lang_adapter_name = "dummy_LA_en"
            model.add_adapter(lang_adapter_name, config=lang_adapter_config)
        else:
            lang_adapter_name = None
            
        # resolve the adapter config
        adapter_config = AdapterConfig.load(adapter_args.adapter_config, **adapter_config_kwargs)
        print(adapter_config)
        # load a pre-trained from Hub if specified
        # note: this logic has changed in versions > 3.1.0: adapter is also loaded if it already exists
        if adapter_args.load_adapter:
            model.load_adapter(
                adapter_args.load_adapter,
                config=adapter_config,
                load_as=adapter_name,
                **adapter_load_kwargs,
            )
        # otherwise, if adapter does not exist, add it
        elif adapter_name not in model.adapters_config:
            model.add_adapter(adapter_name, config=adapter_config)
            #model.add_causal_lm_head(adapter_name)
        # Freeze all model weights except of those of this adapter
        model.train_adapter([adapter_name])
        # Set the adapters to be used in every forward pass
        if lang_adapter_name or use_la_dummy_adapter:
            model.set_active_adapters(Stack(lang_adapter_name, adapter_name))
            #model.active_adapters = Stack(lang_adapter_name, adapter_name)
        else:
            model.set_active_adapters(adapter_name)

        return adapter_name, lang_adapter_name
    
    elif fusion:
        assert isinstance(fuse_langs, list)
        adapter_names = []
        for lang in fuse_langs:
            la_name = f"Llama-2-7b-hf_cc100_{lang}_12500"
            adapter_names.append(la_name)
            la_path =  f"/netscratch/jschlenker/outputs/{la_name}/{la_name}"
            
            la_config = f"/netscratch/jschlenker/outputs/{la_name}/{la_name}/adapter_config.json"
            lang_adapter_config = AdapterConfig.load(la_config)
            model.load_adapter(
                        la_path,
                        config=lang_adapter_config,
                        with_head=False,
                    )

        model.add_adapter_fusion(adapter_names, set_active=True)
        model.train_adapter_fusion([adapter_names])
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode.Use --train_adapter to enable adapter training"
            )

        return None, None

def setup_adapter_training__(
    model,
    adapter_args: AdapterArguments,
    adapter_name: str,
    adapter_config_kwargs: Optional[dict] = None,
    adapter_load_kwargs: Optional[dict] = None,
):
    """Setup model for adapter training based on given adapter arguments.

    Args:
        model (_type_): The model instance to be trained.
        adapter_args (AdapterArguments): The adapter arguments used for configuration.
        adapter_name (str): The name of the adapter to be added.

    Returns:
        Tuple[str, str]: A tuple containing the names of the loaded adapters.
    """
    if adapter_config_kwargs is None:
        adapter_config_kwargs = {}
    if adapter_load_kwargs is None:
        adapter_load_kwargs = {}
    # Setup adapters
    if adapter_args.train_adapter:
        # resolve the adapter config
        adapter_config = AdapterConfig.load(adapter_args.adapter_config, **adapter_config_kwargs)
        # load a pre-trained from Hub if specified
        # note: this logic has changed in versions > 3.1.0: adapter is also loaded if it already exists
        if adapter_args.load_adapter:
            model.load_adapter(
                adapter_args.load_adapter,
                config=adapter_config,
                load_as=adapter_name,
                **adapter_load_kwargs,
            )
        # otherwise, if adapter does not exist, add it
        elif adapter_name not in model.adapters_config:
            model.add_adapter(adapter_name, config=adapter_config)
            print(adapter_config)

        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(adapter_args.lang_adapter_config, **adapter_config_kwargs)
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                **adapter_load_kwargs,
            )
        else:
            lang_adapter_name = None
        # Freeze all model weights except of those of this adapter
        model.train_adapter(adapter_name)
        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters(Stack(lang_adapter_name, adapter_name))
        else:
            model.set_active_adapters(adapter_name)

        return adapter_name, lang_adapter_name
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode.Use --train_adapter to enable adapter training"
            )

        return None, None

def setup_adapter_training(
    model,
    adapter_args: AdapterArguments,
    adapter_name: str,
    adapter_config_kwargs: Optional[dict] = None,
    adapter_load_kwargs: Optional[dict] = None,
):
    """Setup model for adapter training based on given adapter arguments.

    Args:
        model (_type_): The model instance to be trained.
        adapter_args (AdapterArguments): The adapter arguments used for configuration.
        adapter_name (str): The name of the adapter to be added.

    Returns:
        Tuple[str, str]: A tuple containing the names of the loaded adapters.
    """
    if adapter_config_kwargs is None:
        adapter_config_kwargs = {}
    if adapter_load_kwargs is None:
        adapter_load_kwargs = {}
    # Setup adapters
    if adapter_args.train_adapter:
        # resolve the adapter config
        adapter_config = AdapterConfig.load(adapter_args.adapter_config, **adapter_config_kwargs)
        # load a pre-trained from Hub if specified
        # note: this logic has changed in versions > 3.1.0: adapter is also loaded if it already exists
        if adapter_args.load_adapter:
            model.load_adapter(
                adapter_args.load_adapter,
                config=adapter_config,
                load_as=adapter_name,
                **adapter_load_kwargs,
            )
        # otherwise, if adapter does not exist, add it
        elif adapter_name not in model.adapters_config:
            model.add_adapter(adapter_name, config=adapter_config)
            print(adapter_config)
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(adapter_args.lang_adapter_config, **adapter_config_kwargs)
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                **adapter_load_kwargs,
            )
        else:
            lang_adapter_name = None
        # Freeze all model weights except of those of this adapter
        model.train_adapter([adapter_name])
        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters(Stack(lang_adapter_name, adapter_name))
        else:
            model.set_active_adapters(adapter_name)

        return adapter_name, lang_adapter_name
    
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode.Use --train_adapter to enable adapter training"
            )

        return None, None
