from chexpert import CheXpertDataset

ds = CheXpertDataset(config_path='src/config.YAML')
ds.create_local_database('chexpert')