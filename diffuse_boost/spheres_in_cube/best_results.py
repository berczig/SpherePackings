from diffuse_boost import cfg

def load_best_results() -> dict:
    """
    Returns a dict {sphere_num : best_known_box_size}
    Sphere radius of 1
    """
    results = {}
    filename = cfg.get("best_results", "filename")
    with open(filename) as f:
        # first 2 lines are header
        lineID = 0
        while True:
            line = f.readline() 
            if line == "":
                break
            if lineID > 1:
                vals = line.strip().split()
                sphere_num = int(vals[0])
                box_size = float(vals[1])
                results[sphere_num] = box_size
            lineID += 1
    return results
            