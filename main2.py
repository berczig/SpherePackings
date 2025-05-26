from SP.animation import main
from SP.data_generation import get_data_loader
import os
import SP

#main()
loader = get_data_loader(batch_size=10, dataset_path=os.path.join(SP.reffolder, "output/push_simulation/2025-05-12 13_17_55/dataset.pt"))
for k in loader:
    print(k.shape)