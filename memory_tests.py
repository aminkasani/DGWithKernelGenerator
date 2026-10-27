def evaluate_memory(folds, root_dir, TransformTrain,TransformTest, batch_size, num_epochs,useTaa,test_version):
    
    # main_model_name= "AlMainModelV2VarWeight"
    # model = AlMainModelV2VarWeight(num_classes=7)
            

    main_model_name= "AlMobileNetV2"
    model = AlMobileNetV2(num_classes=7)

    # main_model_name= "AlModelV2ArConvRaw"
    # main_model_name= "AlMainModelV2ArConvRawVarWeight"
    
    # model = AlModelV2ArConvRaw(num_classes=5,arconvKernel=7)
    # model = AlMainModelV2ArConvRawVarWeight(num_classes=5)
    # model = AlModelV2ArConvRaw(num_classes=5,arconvKernel=7)
    
    warmup_batches = 5
    measure_batches = 30

    import time
    use_amp=True

    gpu = torch.device(device)


    test_fold = next(iter(folds))
    model_name = main_model_name + "Seed1"

    _, test_dataset = get_train_test_data(
        folds, test_fold, root_dir,
        TransformTest, TransformTest
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=12,
        drop_last=True
    )

    checkpoint_path = (
        "Models/wbc-" + model_name + "-" +
        test_fold + "-" + test_version + ".pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
    )

    model = model.to(device=gpu, dtype=torch.float32)

    torch.backends.cudnn.benchmark = True
    model.eval()

    peak_batches = []
    batch_times = []
    measured_images = 0

    with torch.no_grad():

        for i, (cpu_inputs, _) in enumerate(test_loader):

            if i >= warmup_batches + measure_batches:
                break

            is_measured = i >= warmup_batches

            torch.cuda.synchronize(gpu)

            if is_measured:
                torch.cuda.reset_peak_memory_stats(gpu)
                start_time = time.perf_counter()

            inputs = cpu_inputs.to(
                gpu, non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp
            ):
                outputs = model(inputs)

            torch.cuda.synchronize(gpu)

            if is_measured:

                elapsed = time.perf_counter() - start_time

                peak_mib = (
                    torch.cuda.max_memory_allocated(gpu)
                    / (1024 ** 2)
                )

                batch_times.append(elapsed)
                peak_batches.append(peak_mib)

                measured_images += cpu_inputs.size(0)

            del outputs, inputs

    if not batch_times:
        raise RuntimeError("No measured batches")

    total_time = sum(batch_times)

    throughput = measured_images / total_time

    mean_batch_time_ms = (
        total_time / len(batch_times)
    ) * 1000

    result = {
        "model": main_model_name,
        "dataset": test_fold,
        "seed": 1,
        "batch_size": batch_size,
        "precision": "AMP-FP16" if use_amp else "FP32",
        "measured_batches": len(batch_times),
        "peak_memory_mib": max(peak_batches),
        "mean_batch_peak_mib": np.mean(peak_batches),
        "mean_batch_time_ms": mean_batch_time_ms,
        "throughput_img_s": throughput
    }

    print("\n========== BENCHMARK ==========")

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("===============================")


    exit()
def evaluate_memory_tta(folds, root_dir, TransformTrain,TransformTest, batch_size, num_epochs,useTaa,test_version):
    # Only the first dataset and Seed 1
    use_amp=True
    warmup_batches = 5
    measure_batches = 30
    
    # main_model_name= "AlMainModelV2VarWeight"
    # model = AlMainModelV2VarWeight(num_classes=7)
    
    main_model_name= "AlMainModelV2ArConvRawVarWeight"
    model = AlMainModelV2ArConvRawVarWeight(num_classes=5)
    
            

    # main_model_name= "AlMobileNetV2"

    # main_model_name= "AlModelV2ArConvRaw"
    
    # model = AlMobileNetV2(num_classes=7)
    # model = AlModelV2ArConvRaw(num_classes=5,arconvKernel=7)
    # model = AlModelV2ArConvRaw(num_classes=5,arconvKernel=7)
    
    
    
    test_fold = next(iter(folds))
    model_name = main_model_name + "Seed1"


    gpu = torch.device(device)

    print(f"\n### TTA MEMORY + THROUGHPUT")
    print(f"Model: {model_name}")
    print(f"Dataset: {test_fold}")
    print(f"Batch size: {batch_size}")
    print(f"AMP: {use_amp}")

    _, test_dataset = get_train_test_data(
        folds, test_fold, root_dir,
        TransformTest, TransformTest
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=12,
        drop_last=True
    )

    checkpoint_path = (
        "Models/wbc-" + model_name + "-" +
        test_fold + "-" + test_version + ".pth"
    )

    # Keep the checkpoint on GPU, as in your TTA code
    original_state = torch.load(
        checkpoint_path,
        map_location=gpu,
        weights_only=True
    )

    model = model.to(device=gpu, dtype=torch.float32)
    model.load_state_dict(original_state)

    torch.backends.cudnn.benchmark = True

    # Freeze backbone except the first convolution block
    model.model.requires_grad_(False)
    model.model.features[0].requires_grad_(True)
    model.modelW.requires_grad_(True)

    params = (
        list(model.modelW.parameters()) +
        list(model.model.features[0].parameters())
    )

    optimizerTaa = torch.optim.Adam(
        params,
        lr=0.002
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Retain GradScaler behavior from your original code
    scaler = torch.amp.GradScaler("cuda")

    model.eval()

    peak_allocated = []
    peak_reserved = []
    batch_times = []

    measured_images = 0
    adaptation_batches = 0
    total_batches = warmup_batches + measure_batches

    def run_tta_batch(cpu_inputs):

        # Input transfer is included in timing
        inputs = cpu_inputs.to(
            gpu, non_blocking=True
        )

        # Episodic reset
        model.load_state_dict(original_state)
        optimizerTaa.zero_grad(set_to_none=True)
        optimizerTaa.state.clear()
        model.eval()

        # First forward: gradients are required
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):
            outputs, pooled_features = model(inputs)

            final_layer = model.model.classifier[-1]

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
                temperature=1.0
            )

        valid_mask = pseudo.valid_mask
        has_valid = bool(valid_mask.any().item())

        # Adaptation step
        if has_valid:

            model.train()
            model.model.requires_grad_(False)
            model.model.eval()
            model.model.features[0].requires_grad_(True)

            loss = criterion(
                outputs[valid_mask],
                pseudo.pseudo_labels[valid_mask]
            )

            scaler.scale(loss).backward()
            scaler.step(optimizerTaa)
            scaler.update()

        # Final prediction on the same batch
        model.eval()

        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp
            ):
                final_outputs, _ = model(inputs)
                predicted = final_outputs.argmax(dim=1)

        return has_valid

    for i, (cpu_inputs, _) in enumerate(test_loader):

        if i >= total_batches:
            break

        is_measured = i >= warmup_batches

        torch.cuda.synchronize(gpu)

        if is_measured:
            # Reset peak before the full TTA cycle
            torch.cuda.reset_peak_memory_stats(gpu)
            start_time = time.perf_counter()

        did_adapt = run_tta_batch(cpu_inputs)

        # Wait for all CUDA operations to finish
        torch.cuda.synchronize(gpu)

        if is_measured:

            elapsed = time.perf_counter() - start_time

            allocated_mib = (
                torch.cuda.max_memory_allocated(gpu)
                / (1024 ** 2)
            )

            reserved_mib = (
                torch.cuda.max_memory_reserved(gpu)
                / (1024 ** 2)
            )

            batch_times.append(elapsed)
            peak_allocated.append(allocated_mib)
            peak_reserved.append(reserved_mib)

            measured_images += cpu_inputs.size(0)

            if did_adapt:
                adaptation_batches += 1

    if not batch_times:
        raise RuntimeError(
            "No batches were measured."
        )

    total_time = sum(batch_times)

    throughput = measured_images / total_time
    mean_batch_time_ms = (
        total_time / len(batch_times)
    ) * 1000

    peak_memory = max(peak_allocated)
    mean_peak_memory = np.mean(peak_allocated)

    result = {
        "model": main_model_name + "+TTA",
        "dataset": test_fold,
        "seed": 1,
        "batch_size": batch_size,
        "precision": "AMP-FP16" if use_amp else "FP32",
        "measured_batches": len(batch_times),
        "adaptation_batches": adaptation_batches,
        "measured_images": measured_images,
        "peak_memory_mib": peak_memory,
        "mean_batch_peak_mib": mean_peak_memory,
        "peak_reserved_mib": max(peak_reserved),
        "mean_batch_time_ms": mean_batch_time_ms,
        "throughput_img_s": throughput
    }

    print("\n========== TTA BENCHMARK ==========")

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("===================================")

    exit()
