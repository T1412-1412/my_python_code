from utils.path_tool import get_abs_path
import yaml

def agent_config():
    abs_path = get_abs_path("data_config/agent_config.yaml")
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


agent_conf = agent_config()