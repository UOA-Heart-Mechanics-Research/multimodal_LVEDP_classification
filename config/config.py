from types import SimpleNamespace
CONFIG = SimpleNamespace()  # this will always be live across imports

def load_config(yaml_path):
    """
    Load configuration from a YAML file and update the global CONFIG object.

    Args:
        yaml_path (str): Path to the YAML configuration file.
    """
    import yaml

    # Helper function to recursively convert dictionaries to SimpleNamespace
    def dict_to_ns(d):
        return SimpleNamespace(**{
            k: dict_to_ns(v) if isinstance(v, dict) else v for k, v in d.items()
        })

    # Read and parse the YAML file
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        new_config = dict_to_ns(data)

    # Update the global CONFIG object in-place so all imports see the update
    CONFIG.__dict__.update(new_config.__dict__)


def save_config_and_args(args, file_path):
    """
    Save the current CONFIG object and additional arguments to a YAML file.

    Args:
        args (object): Additional arguments to save (namespace).
        file_path (str): Path to the output YAML file.
    """
    import os
    import yaml

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Helper function to recursively convert SimpleNamespace to dictionaries
    def ns_to_dict(ns):
        return {k: ns_to_dict(v) if isinstance(v, SimpleNamespace) else v for k, v in ns.__dict__.items()}

    # Combine CONFIG and args into a single dictionary
    combined_data = {k: ns_to_dict(v) if isinstance(v, SimpleNamespace) else v 
                     for k, v in CONFIG.__dict__.items() if not k.startswith('_')}
    combined_data['args'] = vars(args) if hasattr(args, '__dict__') else args

    # Save the combined data to a YAML file
    with open(file_path, 'w') as f:
        yaml.dump(combined_data, f)

