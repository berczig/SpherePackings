import torch
import collections

FILE_STRUCTURE = {
    "model":{
        "metadata":{}, # Models primarily need metadata.type = "model"
        "epoch":0,
        "state_dict":{},
        "model_parameters":[],
        "optimizer":None,
    },
    "data":{
        "metadata":{
            "n_packings": {"display":True, "display_label":"Packings"},
            "n_spheres_per_packing": {"display":True, "display_label":"Spheres"},
            "dimension": {"display":True, "display_label":"Dimension"},
        },
        "tensor":{}, # Data type expects a top-level 'tensor' key
    }
}

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load_model(filepath, model_class, optimizer_class, learning_rate, device, verbose):
    # example usage: load_model(my_model.pt, Transformer, Adam, 0.01)
    file = torch.load(filepath)
    print(f"Loaded file {filepath} of type {type(file)}")

    # is my kind of model
    if isinstance(file, dict):
        if "metadata" in file and file["metadata"]["type"] == "model":
            model_parameters = file["model_parameters"]
            print("model_parameters: ", model_parameters)
            model = model_class(**model_parameters)
            optimizer = optimizer_class(model.parameters(), lr=learning_rate)
            model.load_state_dict(file['state_dict'])
            optimizer.load_state_dict(file['optimizer'])
            optimizer_to(optimizer, device)
            print(f"Loaded {filepath} using the class {model_class} with model parameters: {model_parameters} and optimizer class {optimizer_class}")
            if verbose:
                print("model_parameters: ", model_parameters)
                #print(f"model state_dict: ", file['state_dict'])
                #print(f"optimizer state_dict: ", file['optimizer'])
            # Set new learning rate
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

            return model, optimizer, file
    # elif isinstance(file, collections.OrderedDict)
        

def save_model(savepath, model, optimizer, epoch, model_parameters:dict, metadata={}):
    # example usage: save_model(my_model.pt, model, optimizer, 100, {"hidden_layers"=10, "input_layers"=3})
    metadata["type"] = "model"
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "model_parameters":model_parameters,
        "metadata":metadata
    }
    torch.save(checkpoint, savepath)
    #print(f"##### Save model \n {model} \n to {savepath} with optimizer {optimizer}\n at epoch={epoch} created with model parameters: {model_parameters}\#####")
    print(f"##### Save model to {savepath} at epoch={epoch} created with model parameters: {model_parameters}\#####")
    print(f"Saved with additional metadata: {metadata}")

def load_data(filepath, model_class):
    pass

def save_data(filepath, tensor_data, metadata=None):
    if metadata == None:
        metadata = dict([(attr, "N/A") for attr in FILE_STRUCTURE["data"]["metadata"]])
    file = {"metadata":metadata, "tensor":tensor_data}
    torch.save(file, filepath)
    print(f"Saved data at {filepath}")


if __name__ == "__main__":
    # save data
    #t = torch.ones((2,3))
    #save_data(t, "output/test/test_data.pt")
    pass