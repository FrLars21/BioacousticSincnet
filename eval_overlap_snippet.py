# Validation loop
model.eval()

total_loss = 0
total_frame_error = 0
total_sent_error = 0

with torch.no_grad():
    for val_file in test_data_list:
        label = int(val_file['label'])
        signal = load_audio(datadir / val_file['file'], device, cfg.sample_rate)

        chunk_len = (cfg.cw_len * cfg.sample_rate) // 1000  # chunk length in samples
        chunk_shift = (cfg.cw_shift * cfg.sample_rate) // 1000  # chunk shift in samples

        # Calculate number of chunks, including partial chunks at the end
        num_chunks = max(1, int((signal.shape[0] - chunk_len + chunk_shift) / chunk_shift))
        pout = torch.zeros(num_chunks, cfg.num_classes).to(signal.device)

        # Pad the signal to ensure it's long enough for all chunks
        padded_length = (num_chunks - 1) * chunk_shift + chunk_len
        padded_signal = F.pad(signal, (0, max(0, padded_length - signal.shape[0])))

        # Create chunks efficiently
        chunks = padded_signal.unfold(0, chunk_len, chunk_shift)

        # Process in batches
        for i in range(0, num_chunks, cfg.batch_size):
            batch = chunks[i:min(i+cfg.batch_size, num_chunks)]
            batch = batch.unsqueeze(1)  # Shape becomes [batch_size, 1, chunk_len]
            pout[i:i+batch.shape[0]] = model(batch)

        # Calculate predictions and errors for the entire file
        pred = torch.argmax(pout, dim=1)
        lab = torch.full((num_chunks,), label).to(signal.device)
        loss = F.cross_entropy(pout, lab)
        frame_error = (pred != lab).float().mean()
        
        # Calculate sentence-level prediction
        sentence_pred = torch.argmax(pout.sum(dim=0))
        sentence_error = (sentence_pred != lab[0]).float()

        total_loss += loss.item()
        total_frame_error += frame_error.item()
        total_sent_error += sentence_error.item()

total_loss /= len(test_data_list)
total_frame_error /= len(test_data_list)
total_sent_error /= len(test_data_list)