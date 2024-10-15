from dataclasses import dataclass, field
import toml

@dataclass
class Config:
    """
    A class to handle loading configuration files for different environments.

    Attributes:
        config_dir (str): The directory where configuration files are located. Default is "configs".
    """
    config_dir: str = field(default="configs")

    def load(self):
        """
        Returns:
            dict: A dictionary containing the configuration data read from the file.
        Raises:
            Exception: If the configuration file for the specified environment is not found, raises an exception.
        """
        config_path = f"{self.config_dir}/config.toml"
        try:
            with open(config_path, 'r') as file:
                config_data = toml.load(file)
        except FileNotFoundError:
            raise Exception(f"Config file not found: {config_path}")
        
        return config_data
    