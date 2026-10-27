
def evaluate_seeds_tta(folds, root_dir, TransformTrain,TransformTest, batch_size, num_epochs,useTaa,test_version):
    seeds=5
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # main_model_name= "AlMobileNetV2"
    # main_model_name= "AlModelV2ArConvRaw"
    main_model_name= "AlMainModelV2VarWeight"
    # main_model_name= "AlMainModelV2ArConvRawVarWeight"
    
    
    # main_model_name= "AlMobileNetV2SignedSymmetricRank1DW"
    # main_model_name= "AlMainModelV2ArConvRawVarWeight"
    # main_model_name= "AlMobileNetV2PositiveLinearProjectedSymmetricRank1DW"
    # main_model_name= "AlMobileNetV2PositiveNormalizedSymmetricDW"
    # model = AlMobileNetV2PositiveNormalizedSymmetricDW(num_classes=5,arconvKernel=3)
    
    # main_model_name= "AlMobileNetV2PositiveNormalizedSymmetricDWSmoothed"
    print(f"####testing {main_model_name}-TTA####")
    results=[]
    for test_fold in folds:
        train_dataset, test_dataset = get_train_test_data(folds, test_fold, root_dir, TransformTest,TransformTest)
        batch_size=32
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=12)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True, prefetch_factor=12)
        print(f"#{test_fold}:")
        for j in range(seeds):
            print(f"##Seed{j+1}: ",end=" ")
            model_name= main_model_name+f"Seed{j+1}"

            model = AlMainModelV2VarWeight(num_classes=7)
            model = AlMainModelV2ArConvRawVarWeight(num_classes=5)
            # optimizerTaa = torch.optim.Adam(model.modelW.parameters(), lr=0.002)
            params = list(model.modelW.parameters()) + list(model.model.features[0].parameters())
            optimizerTaa = torch.optim.Adam(params, lr=0.01) #was lr=0.002
            torch.backends.cudnn.benchmark = True
            scaler = GradScaler()
            # model_name= "AlMainModelV2VarWeight"
            
            
            original_state  = torch.load("Models/wbc-"+model_name+"-"+test_fold+"-"+test_version+".pth",map_location=device, weights_only=True)
            model = model.to(torch.float32)
            model = model.to(device)

            
            torch.backends.cudnn.benchmark = True

            for param in model.model.parameters():
                param.requires_grad = False
            model.model.features[0].requires_grad_(True)
            model.eval()
            
            all_preds = []
            all_labels = []
            
            train_all_preds = []
            train_all_labels = []
            records = {
                "truth": [],
                "raw_pseudo": [],
                "raw_valid": [],
                "filtered_pseudo": [],
                "filtered_valid": [],
                "before_tta": [],
                "after_tta": [],
            }


            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
                model.load_state_dict(original_state)
                optimizerTaa.zero_grad(set_to_none=True)
                optimizerTaa.state.clear()
                model.eval()
                    
                # with autocast(device_type='cuda',dtype=torch.float16):  # Enable mixed precision
                outputs,pooled_features = model(inputs)
                final_layer = model.model.classifier[-1]
                # filtered_features = topConv(features) 
                pseudo = topConvV3.retain_top_quarterV2(
                    features=pooled_features.detach(),
                    classifier_weight=final_layer.weight.detach(),
                    classifier_bias=(
                        final_layer.bias.detach()
                        if final_layer.bias is not None
                        else None
                    ),
                    confidence_threshold=0.90,
                    margin_threshold=0.20,
                    temperature=1.0,
                )
                valid_mask = pseudo.valid_mask
                
                
                
                
                with torch.no_grad():
                    raw_probs = torch.softmax(outputs.detach(), dim=1)
                    raw_conf, raw_pseudo = raw_probs.max(dim=1)
                    raw_valid = raw_conf >= 0.90

                    filtered_pseudo = pseudo.pseudo_labels.detach()
                # valid_mask = raw_valid
                
                records["truth"].extend(labels.cpu().tolist())
                records["raw_pseudo"].extend(raw_pseudo.cpu().tolist())
                records["raw_valid"].extend(raw_valid.cpu().tolist())
                records["filtered_pseudo"].extend(
                    filtered_pseudo.cpu().tolist()
                )
                records["filtered_valid"].extend(
                    valid_mask.cpu().tolist()
                )
                records["before_tta"].extend(
                    outputs.argmax(dim=1).cpu().tolist()
                )
                if valid_mask.any():
                    model.train()
                    model.model.requires_grad_(False)
                    model.model.eval()
                    model.model.features[0].requires_grad_(True)
                    loss = criterion(
                    outputs[valid_mask],
                    pseudo.pseudo_labels[valid_mask],
                    )
                    scaler.scale(loss).backward()
                    scaler.step(optimizerTaa)
                    scaler.update()
                model.eval()
                with torch.no_grad():
                    outputs,_ = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    records["after_tta"].extend(
                        predicted.cpu().tolist()
                    )
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())


            train_all_preds = all_preds
            train_all_labels = all_labels
            
            
            f1_micro = f1_score(all_labels, all_preds, average='micro')
            f1_macro = f1_score(all_labels, all_preds, average='macro')
            
            train_f1_micro = f1_score(train_all_labels, train_all_preds, average='micro')
            train_f1_macro = f1_score(train_all_labels, train_all_preds, average='macro')

            # print(f'TAA:{useTaa} val Accuracy on fold "{test_fold}":')
            print(f"F1_micro:{f1_micro:.4f},F1_macro:{f1_macro:.4f},val_F1_micro:{train_f1_micro:.4f},val_F1_macro:{train_f1_macro:.4f}")
            # print(classification_report(all_labels, all_preds))
            print("RAW LOGITS")
            pseudo_label_report(
                records["truth"],
                records["raw_pseudo"],
                records["raw_valid"],
            )

            print("FILTERED FEATURES")
            pseudo_label_report(
                records["truth"],
                records["filtered_pseudo"],
                records["filtered_valid"],
            )
            print("BEFORE TTA")
            print(classification_report(
                records["truth"],
                records["before_tta"],
                labels=list(range(5)),
                target_names=CLASS_NAMES,
                zero_division=0,
            ))

            print("AFTER TTA")
            print(classification_report(
                records["truth"],
                records["after_tta"],
                labels=list(range(5)),
                target_names=CLASS_NAMES,
                zero_division=0,
            ))

            cm = confusion_matrix(
                records["truth"],
                records["after_tta"],
                labels=list(range(5)),
                normalize="true",
            )

            print("Normalized confusion matrix:")
            print(np.round(cm, 3))
            
            results.append({
                "model": model_name,
                "seed": j,
                "dataset": test_fold,
                "fold": test_fold,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
            })
    df = pd.DataFrame(results)
    print("##stats")

    per_seed = (
    df.groupby(["fold", "seed"])[["f1_micro", "f1_macro"]]
        .mean()
    )

    overall_stats = per_seed.groupby("fold").agg(["mean", "std"])
    overall_stats.loc['mean'] = overall_stats.mean()
    print(overall_stats)



    exit()
