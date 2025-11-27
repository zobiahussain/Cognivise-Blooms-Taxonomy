# Quick Start Guide - Setting Up API Keys

## Easiest Method: Use the Setup Script

1. **Run the setup script:**
   ```bash
   python setup_api_keys.py
   ```

2. **Follow the prompts:**
   - Enter your Gemini API key (get it from https://aistudio.google.com/apikey)
   - Or press Enter to skip if you don't have one

3. **Done!** The script creates a `.env` file with your keys.

## Manual Method: Create .env File

1. **Create a file named `.env`** in the project root:
   ```
   .env
   ```

2. **Add your key:**
   ```
   GEMINI_API_KEY=your-gemini-key-here
   ```

3. **Save the file**

## Where to Get API Key

### Gemini API Key:
1. Visit: https://aistudio.google.com/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy it

## Verify It Works

After setting up, run your app:
```bash
streamlit run webapp/app.py
```

You should see:
```
✓ Loaded .env file from ...
✓ GEMINI_API_KEY found: ...
```

## API Usage

- **Gemini** (default): Fast, good quality, free tier available
- **Local Model**: Falls back to local Qwen model if Gemini API key is not provided

The app uses **Gemini by default** if API key is available, otherwise uses local models.

## Important Security

- **Never commit `.env` to git!**
- The `.env` file should already be in `.gitignore`
- Don't share your API keys

## Troubleshooting

**"API key not provided" error:**
- Make sure `.env` file is in the project root
- Check for typos in key name (GEMINI_API_KEY)
- Restart your terminal/IDE

**Still having issues?**
- Check the full guide: `API_KEYS_SETUP.md`
- Make sure `python-dotenv` is installed: `pip install python-dotenv`





