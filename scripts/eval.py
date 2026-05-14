import os
import sys
# --- 强制抹除潜伏代理的终极补丁 ---
for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    if k in os.environ:
        del os.environ[k]
# ----------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
from agents.base_agent import Agent
import hydra

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    cfg.eval_agent.agent = hydra.compose(config_name="agent/"+cfg.eval_agent.agent, overrides=[]).agent
    cfg.eval_agent.model = hydra.compose(config_name="model/"+cfg.eval_agent.model, overrides=[]).model
    dataset = BaseDataset(cfg.dataset)
    eval_agent = Agent(cfg.eval_agent)
    eval_agent.eval_dataset(dataset)
    
if __name__ == "__main__":
    main()
