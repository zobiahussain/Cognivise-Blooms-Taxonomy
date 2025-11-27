# Model Download Instructions

## Quick Setup

The trained model is required for the application to work. Download it from GitHub Releases.

### Option 1: Download from GitHub Releases (Recommended)

1. **Go to GitHub Releases**
   - Visit: https://github.com/YOUR_USERNAME/YOUR_REPO/releases
   - Find the latest release
   - Download `blooms-taxonomy-model-v1.0.zip`

2. **Extract the Model**
   ```bash
   # Extract the zip file
   unzip blooms-taxonomy-model-v1.0.zip
   
   # The extracted folder should be: final_model/
   ```

3. **Place in Correct Location**
   ```bash
   # Move to webapp/models/ directory
   mv final_model webapp/models/final_model
   ```

4. **Verify Installation**
   - Check that `webapp/models/final_model/` contains:
     - adapter_config.json
     - adapter_model.safetensors
     - tokenizer_config.json
     - tokenizer.json
     - tokenizer.model
     - special_tokens_map.json
     - training_args.bin

### Option 2: Manual Download

If you prefer to download manually:

1. Download `blooms-taxonomy-model-v1.0.zip` from releases
2. Extract to a temporary location
3. Copy the `final_model` folder to `webapp/models/final_model`
4. Ensure all files are present (see list above)

## Model Information

- **Model Name**: Bloom's Taxonomy Classifier
- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Accuracy**: 82.42% on test set
- **Size**: ~21 MB
- **Training Data**: ~2,948 questions

## Verification

After placing the model, verify it works:

```bash
# Run the app
streamlit run webapp/app.py

# Go to "Analyze Exam" page
# Enter a test question: "What is photosynthesis?"
# Click "Analyze Exam"
# Should work without errors
```

## Troubleshooting

**"Model not found" error:**
- Verify the path: `webapp/models/final_model/`
- Check all files are present (8 files total)
- Ensure file permissions allow reading

**"Permission denied" error:**
```bash
chmod -R 755 webapp/models/final_model
```

**Still having issues?**
- Check the model files are not corrupted
- Re-download from releases
- Verify you're using the correct directory structure

