from diffuse_boost import cfg
from diffuse_boost.spheres_in_cube.DiffusionModel_PESC import sample_diffusion_model, SetTransformer, PointNetPlusPlus
from diffuse_boost.spheres_in_cube.FlowMatching_PESC import FlowSetTransformer, sample_flow_model
import torch
import os
from datetime import datetime
from diffuse_boost.spheres_in_cube import data_load_save

def load_model(sec, filepath):
    d = int(sec["dimension"])
    # You have to make sure that the parameters you use to load are the 
    # same you used to save - they won't be saved (e.g. st_dim_hidden won't be saved)
    # Model init
    if sec.get("model_type", "pointnet").lower() == "pointnet":
        model = PointNetPlusPlus(d)
    else:
        model = SetTransformer(
            dim_in=2*d,
            dim_hidden=int(sec.get("st_dim_hidden", 128)),
            num_heads=int(sec.get("st_num_heads", 4)),
            num_inds=int(sec.get("st_num_inds", 16)),
            num_isab=int(sec.get("st_num_isab", 2)),
            dim_out=d
        )
    state_dict = torch.load(filepath)
    print(f"state: {state_dict.keys()}")
    model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    sec = cfg["diffusion_model"]
    sec2 = cfg["diffusion_model_sample"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sphere_radius = float(sec["sphere_radius"])
    print(f"device: {device}")

    # load model
    #model = load_model(sec, sec2["load_model_path"])
    model_type = sec.get("model_type", "pointnet").strip().lower()
    if model_type == "pointnet":
        model_class = PointNetPlusPlus
    elif model_type == "transformer":
        model_class = SetTransformer
    elif model_type == "flowmatch":
        model_class = FlowSetTransformer
    model, optimizer, file = data_load_save.load_model(
        sec2["load_model_path"], 
        model_class,
        torch.optim.AdamW,
        learning_rate=float(sec["learning_rate"]), device=device, verbose=True)
    model.to(device)

    # Sample new packings
    sample_new_points = int(sec.get("sample_new_points", 10))
    sample_new_points_batch_size = int(sec.get("sample_new_points_batch_size", 10))
    num_points = int(sec["num_spheres"])
    d = int(sec["dimension"])
    clip_range = float(sec["clip_sample_range"])
    print(f"Starting diffusion-based generation of {sample_new_points} samples...")

    if model_type in ["pointnet", "transformer"]:
        samples = sample_diffusion_model(
            model,
            sample_new_points,
            sample_new_points_batch_size,
            num_points,
            int(sec["num_train_timesteps"]),
            int(sec["num_inference_timesteps"]),
            float(sec["beta_start"]),
            float(sec["beta_end"]),
            sec.getboolean("clip_sample"),
            clip_range,
            sphere_radius,
            device,
            d, sec["clamping_layer"]
        )
    elif model_type == "flowmatch":
        samples = sample_flow_model(
            model, sample_new_points, sample_new_points_batch_size, num_points,
            device, sphere_radius, 0, clip_range, d
        )

    # Save outputs

    s_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Generated {sample_new_points} samples, saving to {sec2['samples_save_path']}")
    output_save_path = os.path.join(sec["save_generated_path"], f"generated_{s_now}.pt")
    os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
    torch.save(torch.from_numpy(samples), output_save_path)
    print(f"Samples saved to {output_save_path}")