import json
with open('config', 'r') as f:
    config = json.load(f)
with open('config', 'w') as f:
    json.dump(config, f, sort_keys=True, indent=4)
