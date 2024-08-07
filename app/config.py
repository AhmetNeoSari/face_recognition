from dataclasses import dataclass, field
import toml

@dataclass
class Config:
    config_dir: str = field(default="configs")

    def load(self, env: str):
        config_path = f"{self.config_dir}/config.{env}.toml"
        try:
            with open(config_path, 'r') as file:
                config_data = toml.load(file)
        except FileNotFoundError:
            raise Exception(f"Config file for environment '{env}' not found: {config_path}")
        
        return config_data
    