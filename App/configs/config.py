from dataclasses import dataclass, field
import toml

@dataclass
class Config:
    config_path: str = field(default="configs")

    def load(self,):
        with open(self.config_path, 'r') as file:
            config_data = toml.load(file)
        
        return config_data
    
if __name__ == "__main__":
    config = Config("config.local.toml")
    loaded_config = config.load()
    print(loaded_config["tracker"])