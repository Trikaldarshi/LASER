
## freeze the layers except the last 2 layers
if self.config['runner']['freeze_layers']==True:     
    if self.args.upstream == 'hubert_base':
        print("Layer freezing for hubert")
        count = 0
        for name, param in model.named_parameters():
            count += 1
            if (count<(177) and count>2):
                
                param.requires_grad = False
                print(name, param.requires_grad)
            elif 'model.layer_norm' in name:
                param.requires_grad = False
                print(name, param.requires_grad)
            elif 'model.final_proj' in name:
                param.requires_grad = False
                print(name, param.requires_grad)
            else:
                print(name, param.requires_grad)
    elif self.args.upstream == 'wavlm_base':
        print("Layer freezing for wavlm")
        count = 0
        for name, param in model.named_parameters():
            count += 1
            if count < 207 and count > 1:
                param.requires_grad = False
                print(name, param.requires_grad)
            elif 'model.layer_norm' in name:
                param.requires_grad = False
                print(name, param.requires_grad)
            elif 'model.final_proj' in name:
                param.requires_grad = False
                print(name, param.requires_grad)
            else:
                print(name, param.requires_grad)

    ## count the number of trainable parameters, added by amit
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: ", trainable_params)

else:
    if self.config['runner']['baseline'] == 'superb':
        print("The baseline is superb")
    elif self.config['runner']['baseline'] == 'custom':
        print("Evaluation of custom model")
        custom_ckpt_path = self.config['downstream_expert']['datarc']['test_base_path']
        print("The path for the custom finetuned upstream model is:", custom_ckpt_path)
        tuned_ckpt = torch.load(custom_ckpt_path)
        print("Updating the weights of the upstream model with the custom finetuned upstream model")
        model.load_state_dict(tuned_ckpt['Upstream']) # Loading the custom upstream model