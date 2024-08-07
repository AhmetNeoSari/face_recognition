from dataclasses import dataclass, field
import toml

@dataclass
class Config:
    config_path: str = field(default="/home/ahmet/workplace/face_recognition/App/configs/config.toml")

    def load(self):
        with open(self.config_path, 'r') as file:
            config_data = toml.load(file)
        
        combined_config = {
            "detection": config_data['detection'],
            "recognition": config_data['recognition'],
            "tracker": config_data['tracker']
        }
        
        return combined_config
    
if __name__ == "__main__":
    config = Config()
    loaded_config = config.load()
    print(loaded_config["tracker"])